<#
.SYNOPSIS
  Run the organiser's HCMR-EV stack (OpenNG3 + Monadnock + EndoNav browser physics) on Windows
  with Docker Desktop, following HCMR-EV-Integration/HANDOFF.md (no WSL shell needed).

.DESCRIPTION
  Commands:
    start       build images if missing, start containers (plus their reset coordinator), compile +
                launch OpenNG3's ROS graph, wait for joint feedback, re-zero the EndoNav follower, serve
                the web pages and run a fresh physics page in headless Chrome (-Visible: in a normal
                window instead). Then it checks the page end to end (bridge sockets, a checked reset,
                a short guidewire push the page must follow, reset again) and repairs it if needed
                (re-zero + reload, then a new Chrome); exits 1 if the page still does not follow
    status      containers, joint feedback, the latest guidewire setpoint and the physics page
    reset       their coordinated episode reset (/hcmr/episode/reset): robot home, page and noise reset
    test        send 5 mm/s to module 1 (guidewire) for ~4 s and report the robot motion
    rebaseline  re-zero the EndoNav follower (needed after Monadnock restarts); reload the page after
    stop        stop the containers, the local page servers and the physics-page Chrome (images are kept)

  Prerequisites: Docker Desktop running, and the organiser repositories extracted as sibling folders
  in -Root (default ..\HCMR-EV next to this repo): HCMR-EV-OpenNG3, HCMR-EndoNavSimDocker,
  HCMR-EV-Integration, HCMR-EV-RL.

  Local fix applied: the Monadnock image shipped in Hackathon.zip defaults to 4 modules
  (126-byte UDP feedback) while the current OpenNG3 bridge accepts only 2 (66 bytes). The script
  runs it with SIM_MODULE_COUNT=2 through docker-compose.local-2module.yml in -Root.
  The extracted web\meshes entries are git symlink placeholders (text files) on Windows; the script
  replaces them with directory junctions to HCMR-EndoNavSimDocker\meshes so the page finds its scenario.

  Keep exactly one physics page connected. A second one (for example a background browser tab) also
  publishes device states and corrupts measurements. The probe and relay lock onto the page that
  restarted at the last reset, but closing extra tabs is the real fix.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start
#>
param(
    [Parameter(Position = 0)][ValidateSet("start", "status", "test", "reset", "rebaseline", "stop")][string]$Command = "status",
    [string]$Root = "",
    [string]$Python = "",
    [switch]$NoBrowser,
    [switch]$Visible
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
$RepoRoot = Split-Path $ScriptDir -Parent
$RepoPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

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
function Repair-MeshLinks {
    foreach ($web in @((Join-Path $Endo "src\catforce_ros_bridge\web\meshes"), (Join-Path $Endo "src\sensor_sim\web\meshes"))) {
        $item = Get-Item $web -ErrorAction SilentlyContinue
        if ($item -and -not $item.PSIsContainer -and $item.Length -lt 100) {
            Remove-Item $web  # git symlink placeholder (contains "../../../meshes")
            New-Item -ItemType Junction -Path $web -Target (Join-Path $Endo "meshes") | Out-Null
            "  $web -> junction to meshes"
        }
    }
}
function Start-PhysicsPage([bool]$Headless) {
    if (-not (Test-Path $RepoPython)) { "  repo venv not found ($RepoPython); open http://localhost:8000/catheter_guidewire_ros2.html yourself"; return }
    $flag = if ($Headless) { "True" } else { "False" }
    & $RepoPython -c "import sys; sys.path.insert(0, r'$RepoRoot\src'); from stroke_nav.organiser.live.renderer import PhysicsPage; print('  physics page:', PhysicsPage().start(headless=$flag).state()['mesh_info'])"
}
function Invoke-PagePython([string]$Code) {
    & $RepoPython -c "import sys; sys.path.insert(0, r'$RepoRoot\src'); from stroke_nav.organiser.live.renderer import PhysicsPage; p = PhysicsPage(); $Code"
}
function Get-PageInsertionMm {
    $s = Invoke-PagePython "print(p.state()['guidewire_insertion'])" 2>$null
    if ("$s" -match '(-?[\d.]+)\s*mm') { [double]$Matches[1] } else { $null }
}
function Stop-PhysicsChrome {
    # The Chrome that PhysicsPage starts (its own profile). Kill only the root process with its tree:
    # a page left from an earlier run is otherwise reused by the next start.
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -match "chrome-physics-profile" })
    $ids = @($procs | ForEach-Object { $_.ProcessId })
    $roots = @($procs | Where-Object { $ids -notcontains $_.ParentProcessId })
    foreach ($p in $roots) { taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null }
    if ($roots.Count) { Write-Host "  closed the physics-page Chrome (pid $(($roots | ForEach-Object { $_.ProcessId }) -join ', '))" }
}
function Get-JointPositions {
    # /ng3/joint_states positions [m, rad, m, rad], or $null while it is not published
    $out = Invoke-InContainer $Bridge @"
$RosSrc
timeout 10 ros2 topic echo /ng3/joint_states --once --qos-reliability best_effort --field position 2>/dev/null | grep -v -e lost -e 'total count' | head -1
"@
    if ("$out" -match '\[([-\d.e, ]+)\]') { [double[]]($Matches[1] -split ',\s*') } else { $null }
}
function Wait-JointFeedback([int]$TimeoutS = 90) {
    # The follower latches its zero from the first joint state; re-zeroing before feedback flows
    # leaves the page pinned at absolute positions.
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        $pos = Get-JointPositions
        if ($pos) { Write-Host "  joint feedback [m, rad]: $($pos -join ', ')"; return $true }
        Start-Sleep -Seconds 2
    }
    Write-Host "  no /ng3/joint_states after $TimeoutS s (see /tmp/full_system.log in openng3-sim)"
    return $false
}
function Invoke-CheckedReset {
    # "" on success, otherwise the reset coordinator's message
    $out = Invoke-InContainer $Bridge @"
$RosSrc
timeout 40 ros2 service call /hcmr/episode/reset std_srvs/srv/Trigger '{}' 2>&1 | grep -o "success=[A-Za-z]*, message='[^']*'"
"@
    if ("$out" -match 'success=True') { "" } elseif ("$out") { "$out" } else { "no reply from /hcmr/episode/reset" }
}
function Test-PhysicsPage {
    # "" when the page is connected and follows the robot, otherwise what is wrong
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $conns = @(Get-NetTCPConnection -State Established -RemotePort 18765, 18766, 18767, 18768 -ErrorAction SilentlyContinue)
        if ($conns.Count -ge 4) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    # Chrome's sockets belong to its network-service process, which carries our profile on its command line
    $foreign = @($conns | ForEach-Object { $_.OwningProcess } | Select-Object -Unique | ForEach-Object {
            $p = Get-CimInstance Win32_Process -Filter "ProcessId=$_"
            if ($p -and $p.CommandLine -notmatch "chrome-physics-profile") { "$($p.Name) (pid $_)" } })
    if ($foreign.Count) { return "FOREIGN: another browser holds bridge sockets: $($foreign -join ', '). Close its catheter_guidewire_ros2.html tab; it steals the robot commands" }
    if ($conns.Count -lt 4) { return "only $($conns.Count) of 4 bridge sockets connected" }
    $err = Invoke-CheckedReset
    if ($err) { return "coordinated reset failed: $err" }
    $ins = $null
    $deadline = (Get-Date).AddSeconds(10)
    do { $ins = Get-PageInsertionMm; if ($null -ne $ins -and [math]::Abs($ins) -lt 2) { break }; Start-Sleep -Seconds 1 } while ((Get-Date) -lt $deadline)
    if ($null -eq $ins -or [math]::Abs($ins) -ge 2) { return "page guidewire at '$ins' mm after the reset instead of 0 (follower zero latched wrong?)" }
    $before = Get-JointPositions
    Invoke-InContainer "openng3-sim" @"
$RosSrc
timeout 6 ros2 topic pub -r 20 /control/rqt/module_1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 5.0}}' > /dev/null 2>&1
timeout 1.5 ros2 topic pub -r 20 /control/rqt/module_1/cmd_vel geometry_msgs/msg/Twist '{}' > /dev/null 2>&1
"@ | Out-Null
    $after = Get-JointPositions
    if (-not $before -or -not $after) { return "no joint feedback around the test push" }
    $robot = ($after[0] - $before[0]) * 1000.0
    if ($robot -lt 5) { return "robot module 1 moved only $([math]::Round($robot, 1)) mm for 5 mm/s over 6 s (OpenNG3 command path)" }
    $deadline = (Get-Date).AddSeconds(15)  # the page may run slower than real time
    do {
        $ins = Get-PageInsertionMm
        if ($null -ne $ins -and $ins -ge 0.5 * $robot -and $ins -le $robot + 10) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    $msg = "robot pushed $([math]::Round($robot, 1)) mm, page guidewire at '$ins' mm"
    $err = Invoke-CheckedReset
    if ($null -eq $ins -or $ins -lt 0.5 * $robot -or $ins -gt $robot + 10) { return "page does not follow: $msg" }
    if ($err) { return "reset after the test push failed: $err" }
    Write-Host "  page follows the robot ($msg); reset to home"
    return ""
}
function Confirm-PhysicsPage([bool]$Headless) {
    # check, then repair: re-zero the follower and reload the page, then a fresh Chrome
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $problem = Test-PhysicsPage
        if (-not $problem) { return $true }
        Write-Host "  check $attempt failed: $problem"
        if ($problem -like "FOREIGN:*") { return $false }  # not ours to repair
        if ($attempt -eq 1) {
            Write-Host "  repair: re-zero the follower and reload the page"
            Invoke-InContainer $Bridge $ReBaseline | Out-Null
            Invoke-PagePython "p.reload(); print('  page reloaded:', p.state()['mesh_info'])" | Out-Host
        } elseif ($attempt -eq 2) {
            Write-Host "  repair: re-zero the follower and start a new physics-page Chrome"
            Stop-PhysicsChrome
            Start-Sleep -Seconds 2
            Invoke-InContainer $Bridge $ReBaseline | Out-Null
            Start-PhysicsPage $Headless | Out-Host
        }
    }
    return $false
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
    for ($i = 0; $i -lt 20 -and -not (Test-Port $Port); $i++) { Start-Sleep -Milliseconds 500 }  # Chrome must not load it early
    "  page server :$Port -> $Dir$(if (-not (Test-Port $Port)) { '  (not answering yet)' })"
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
        Repair-MeshLinks
        Invoke-EndoCompose up --no-build -d ros2-bridge noise-injection
        Invoke-OpenNG3Compose up --no-build -d openng3-sim monadnock-sim
        docker compose -f (Join-Path $Integration "docker-compose.reset.yml") up -d --build
        "== OpenNG3 build + launch"
        Invoke-InContainer "openng3-sim" @"
$RosSrc
cd /OpenNG3
colcon build --packages-up-to bridge --cmake-args -DAMENT_CMAKE_SYMLINK_INSTALL=OFF > /tmp/colcon.log 2>&1 || { tail -20 /tmp/colcon.log; exit 1; }
if pgrep -f "ros2 launch bridge full_system.launch.py" > /dev/null; then echo "  ROS graph already running"; else
  nohup bash -lc "source /opt/ros/jazzy/setup.bash; source /OpenNG3/install/setup.bash; export ROS_DOMAIN_ID=42; ros2 launch bridge full_system.launch.py network_mode:=docker pim_mode:=none rqt_dashboard_mode:=off" > /tmp/full_system.log 2>&1 &
  echo "  ROS graph launched (log: /tmp/full_system.log in openng3-sim)"; fi
"@
        "== joint feedback"
        Start-Sleep -Seconds 5
        if (-not (Wait-JointFeedback)) { "Robot feedback missing; not starting the physics page."; exit 1 }
        "== follower re-zero"
        Invoke-InContainer $Bridge $ReBaseline
        "== web pages"
        Start-PageServer 8000 (Join-Path $Endo "src\catforce_ros_bridge\web")
        Start-PageServer 8001 (Join-Path $Endo "src\sensor_sim\web")
        $cors = Join-Path $ScriptDir "cors_static_server.py"
        Start-PageServer 8002 $Scenario $cors
        "== physics page (loads the cas-reference scenario by default)"
        if (-not $NoBrowser) {
            Stop-PhysicsChrome  # never reuse a page from an earlier run
            Start-PhysicsPage (-not $Visible)
            "== physics page check"
            if (-not (Confirm-PhysicsPage (-not $Visible))) {
                "The physics page does not follow the robot after repairs. Check 'status', close any other"
                "catheter_guidewire_ros2.html tab, then run 'stop' and 'start' again."
                exit 1
            }
        }
        "Noisy viewer:   http://localhost:8001/noisy_visualization_ros2.html  (view only; does not run physics)"
        "Scenario files: http://localhost:8002/vessel.obj and /goals.txt"
        "Do not open catheter_guidewire_ros2.html in another browser tab: a second physics page also publishes."
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
        $conns = @(Get-NetTCPConnection -State Established -RemotePort 18765 -ErrorAction SilentlyContinue)
        $owners = @($conns | ForEach-Object { (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName } | Select-Object -Unique)
        "  browsers connected to the physics bridge: $($conns.Count) page(s) from $($owners -join ', ')  (should be 1)"
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
    "reset" {
        Invoke-InContainer $Bridge @"
$RosSrc
timeout 40 ros2 service call /hcmr/episode/reset std_srvs/srv/Trigger '{}' 2>&1 | grep -o "message='[^']*'"
"@
    }
    "rebaseline" { Invoke-InContainer $Bridge $ReBaseline; "Reload the physics page now so both rods restart from zero." }
    "stop" {
        Invoke-OpenNG3Compose stop openng3-sim monadnock-sim
        Invoke-EndoCompose stop ros2-bridge noise-injection
        docker stop hcmr-reset-coordinator 2>$null | Out-Null
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "http\.server 800[01]|cors_static_server\.py" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Stop-PhysicsChrome
        "Stopped containers, page servers and the physics-page Chrome (images and the hcmr-ev-integration network are kept)."
    }
}
