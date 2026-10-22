"""Tools that talk to the organiser's live stack (running locally via tools/organiser_stack.ps1).

* ``renderer``   start their physics page in headless Chrome, with the flags their RL renderer
                 uses, and read its state over the Chrome DevTools Protocol
* ``probe``      play an action sequence on their robot and record ground-truth device states
                 (``probe_ros.py`` runs inside the EndoNav container, which has their message types)
* ``bridge``     closed-loop access: ``LiveBridge`` keeps ``relay_ros.py`` running in the container and
                 exchanges reset / step(action) / observation as JSON lines
"""
