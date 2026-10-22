<#
.SYNOPSIS
  Run the organiser's HCMR-EV stack (OpenNG3 + Monadnock + EndoNav browser physics) on Windows
  with Docker Desktop, following HCMR-EV-Integration/HANDOFF.md (no WSL shell needed).

.DESCRIPTION
  Commands:
    start       build images if missing, start containers, compile + launch OpenNG3's ROS graph,
                re-zero the EndoNav follower, serve the web pages and open the physics page
    status      containers, joint feedback and the latest guidewire setpoint
    test        send 5 mm/s to module 1 (guidewire) for ~4 s and report the robot motion
    rebaseline  re-zero the EndoNav follower (needed after Monadnock restarts); reload the page after
    stop        stop the containers and the local page servers (images are kept)

  Prerequisites: Docker Desktop running, and the organiser repositories extracted as sibling folders
  in -Root (default ..\HCMR-EV next to this repo): HCMR-EV-OpenNG3, HCMR-EndoNavSimDocker,
  HCMR-EV-Integration, HCMR-EV-RL.

  Local fix applied: the Monadnock image shipped in Hackathon.zip defaults to 4 modules
  (126-byte UDP feedback) while the current OpenNG3 bridge accepts only 2 (66 bytes). The script
  runs it with SIM_MODULE_COUNT=2 through docker-compose.local-2module.yml in -Root.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start
#>
param(
    [Parameter(Position = 0)][ValidateSet("start", "status", "test", "rebaseline", "stop")][string]$Command = "status",
    [string]$Root = "",
    [string]$Python = "",
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path $ScriptDir -Parent) "..\HCMR-EV" }
$Root = (Resolve-Path $Root).Path
# Docker and Compose report progress on stderr; don't let that abort the script (exit codes are what matter).
$ErrorActionPreference = "Continue"
$OpenNG3 = Join-Path $Root "HCMR-EV-OpenNG3"
$Endo = Join-Path $Root "HCMR-EndoNavSimDocker"
$Integration = Join-Path $Root "HCMR-EV-Integration"
$Scenario = Join-Path $Root "HCMR-EV-RL\scenarios\cas-reference"
$ComposeDir = Join-Path $OpenNG3 "docker\compose"
$Override = Join-Path $Root "docker-compose.local-2module.yml"
$Bridge = "hcmr-endonavsimdocker-ros2-bridge-1"
$RosSrc = "source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=42"
if (-not $Python) { $Python = (Get-Command python -ErrorAction SilentlyContinue).Source }

foreach ($p in @($OpenNG3, $Endo, $Integration)) { if (-not (Test-Path $p)) { throw "missing $p (extract the organiser repositories into $Root)" } }
if (-not (Test-Path $Override)) {
    @(
        "# Local override: the shipped Monadnock image defaults to 4 modules; OpenNG3 accepts 2.",
        "services:", "  monadnock-sim:", "    environment:", '      SIM_MODULE_COUNT: "2"'
    ) | Set-Content -Encoding ascii $Override
}

function Set-OpenNG3Env {
    $env:HOME = $HOME; $env:ROS2WS_PATH = $OpenNG3; $env:HCMR_INTEGRATION_NETWORK = "hcmr-ev-integration"
    $env:COMPOSE_NETWORK_NAME = "ros-control-net"; $env:CONTROL_NET_SUBNET = "172.30.0.0/16"; $env:CONTROL_NET_GATEWAY = "172.30.0.1"
    $env:ROS_DOMAIN_ID = "42"; $env:STEVE_CMD = "true"; $env:PHILLIP_CMD = "true"
}
function Invoke-EndoCompose { docker compose --project-directory $Endo -f (Join-Path $Endo "docker-compose.yml") -f (Join-Path $Endo "docker-compose.integration.yml") @args }
function Invoke-OpenNG3Compose {
    Set-OpenNG3Env
    docker compose --project-directory $ComposeDir -f (Join-Path $ComposeDir "docker-compose.multi-service.yml") `
        -f (Join-Path $ComposeDir "docker-compose.windows.yml") -f (Join-Path $ComposeDir "docker-compose.integration.yml") `
        -f (Join-Path $Integration "docker-compose.monadnock-image.yml") -f $Override --profile monadnock @args
}
function Invoke-InContainer([string]$Container, [string]$Script) {
    $tmp = Join-Path $env:TEMP "stroke_nav_stack_$([guid]::NewGuid().ToString('N')).sh"
    ($Script -replace "`r`n", "`n") | Set-Content -NoNewline -Encoding ascii $tmp
    docker cp $tmp "${Container}:/tmp/stroke_nav_cmd.sh" | Out-Null
    Remove-Item $tmp
    docker exec $Container bash /tmp/stroke_nav_cmd.sh
}
function Test-Port([int]$Port) { try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Port/").StatusCode -eq 200 } catch { $false } }
function Start-PageServer([int]$Port, [string]$Dir, [string]$Script = "") {
    if (Test-Port $Port) { "  page server :$Port already running"; return }
    if (-not $Python) { throw "python not found; pass -Python <path>" }
    $pyArgs = if ($Script) { @("-P", $Script, "$Port", $Dir) } else { @("-m", "http.server", "$Port", "--bind", "127.0.0.1", "--directory", $Dir) }
    Start-Process -FilePath $Python -ArgumentList $pyArgs -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 1
    "  page server :$Port -> $Dir"
}

$ReBaseline = @"
$RosSrc
source /workspace/install/setup.bash
timeout 20 ros2 service call /endovascular/follower/enabled std_srvs/srv/SetBool "{data: false}" > /dev/null 2>&1
sleep 1
timeout 20 ros2 service call /endovascular/follower/enabled std_srvs/srv/SetBool "{data: true}" 2>&1 | grep -o "message='[^']*'"
"@

switch ($Command) {
    "start" {
        "== images"
        $have = docker images --format "{{.Repository}}:{{.Tag}}"
        if ($have -notcontains "hcmr/monadnock-sim:dev") {
            $archive = Join-Path $OpenNG3 "docker\archive\monadnock-model-latest"
            $tar = Join-Path $env:TEMP "monadnock-model-latest.tar"
            if (Test-Path $tar) { Remove-Item $tar }
            tar -cf $tar -C $archive .
            docker load --input $tar; docker tag monadnock-model:latest hcmr/monadnock-sim:dev; Remove-Item $tar
        }
        if (($have -notcontains "hcmr/endonav-ros2-bridge:dev") -or ($have -notcontains "hcmr/endonav-noise-injection:dev")) { Invoke-EndoCompose build ros2-bridge noise-injection }
        if ($have -notcontains "hcmr/openng3-bridge:dev") { Invoke-OpenNG3Compose build openng3-sim }
        if (-not (docker network ls --format "{{.Name}}" | Where-Object { $_ -eq "hcmr-ev-integration" })) {
            docker network create hcmr-ev-integration | Out-Null
        }
        "== containers"
        Invoke-EndoCompose up --no-build -d ros2-bridge noise-injection
        Invoke-OpenNG3Compose up --no-build -d openng3-sim monadnock-sim
        "== OpenNG3 build + launch"
        Invoke-InContainer "openng3-sim" @"
$RosSrc
cd /OpenNG3
colcon build --packages-up-to bridge --cmake-args -DAMENT_CMAKE_SYMLINK_INSTALL=OFF > /tmp/colcon.log 2>&1 || { tail -20 /tmp/colcon.log; exit 1; }
if pgrep -f "ros2 launch bridge full_system.launch.py" > /dev/null; then echo "  ROS graph already running"; else
  nohup bash -lc "source /opt/ros/jazzy/setup.bash; source /OpenNG3/install/setup.bash; export ROS_DOMAIN_ID=42; ros2 launch bridge full_system.launch.py network_mode:=docker pim_mode:=none rqt_dashboard_mode:=off" > /tmp/full_system.log 2>&1 &
  echo "  ROS graph launched (log: /tmp/full_system.log in openng3-sim)"; fi
"@
        Start-Sleep -Seconds 12
        "== follower re-zero"
        Invoke-InContainer $Bridge $ReBaseline
        "== web pages"
        Start-PageServer 8000 (Join-Path $Endo "src\catforce_ros_bridge\web")
        Start-PageServer 8001 (Join-Path $Endo "src\sensor_sim\web")
        $cors = Join-Path $ScriptDir "cors_static_server.py"
        Start-PageServer 8002 $Scenario $cors
        "Physics page:   http://localhost:8000/catheter_guidewire_ros2.html  (keep it open, visible)"
        "Noisy viewer:   http://localhost:8001/noisy_visualization_ros2.html"
        "Scenario files: http://localhost:8002/vessel.obj and /goals.txt  (or load them with the page's buttons from $Scenario)"
        if (-not $NoBrowser) { Start-Process "http://localhost:8000/catheter_guidewire_ros2.html" }
    }
    "status" {
        "== containers"; docker ps -a --filter "name=openng3-sim" --filter "name=monadnock-sim" --filter "name=hcmr-endonavsimdocker" --format "{{.Names}}  {{.Status}}"
        "== robot joints and guidewire setpoint"
        Invoke-InContainer $Bridge @"
$RosSrc
source /workspace/install/setup.bash
echo "  joint positions [m, rad]: `$(timeout 15 ros2 topic echo /ng3/joint_states --once --qos-reliability best_effort --field position 2>/dev/null | grep -v -e lost -e 'total count' | head -1)"
echo "  guidewire setpoint [mm]:  `$(timeout 15 ros2 topic echo /endovascular/guidewire/command --once --qos-reliability best_effort --field insertion_setpoint 2>/dev/null | grep -v -e lost -e 'total count' -e '^---' | head -1)"
"@
        foreach ($p in 8000, 8001, 8002) { "  page server :$p running: $(Test-Port $p)" }
    }
    "test" {
        Invoke-InContainer "openng3-sim" @"
$RosSrc
pos() { timeout 8 ros2 topic echo /ng3/joint_states --once --qos-reliability best_effort --field position 2>/dev/null | grep -v -e lost -e 'total count' | head -1; }
echo "  before: `$(pos)"
timeout 6 ros2 topic pub -r 20 /control/rqt/module_1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 5.0}}' > /dev/null 2>&1
timeout 1.5 ros2 topic pub -r 20 /control/rqt/module_1/cmd_vel geometry_msgs/msg/Twist '{}' > /dev/null 2>&1
echo "  after:  `$(pos)   (module 1 should have advanced by roughly 10-30 mm)"
"@
    }
    "rebaseline" { Invoke-InContainer $Bridge $ReBaseline; "Reload the physics page now so both rods restart from zero." }
    "stop" {
        Invoke-OpenNG3Compose stop openng3-sim monadnock-sim
        Invoke-EndoCompose stop ros2-bridge noise-injection
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "http\.server 800[01]|cors_static_server\.py" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        "Stopped containers and page servers (images and the hcmr-ev-integration network are kept)."
    }
}
