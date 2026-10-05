#!/usr/bin/env python3
"""
Parse an SSH tunnel command and generate a PowerShell script with separate tunnels.

Usage:
    python generate_tunnels.py "ssh -L 8000:node1:8000 -L 8001:node2:8000 user@host"
    
Or interactively:
    python generate_tunnels.py
    > Paste your SSH command
"""

import sys
import re
from pathlib import Path


def parse_ssh_command(ssh_cmd: str) -> tuple[list[dict], str, str]:
    """Parse SSH command and extract tunnel configs."""
    # Extract -L port forwards: -L local_port:remote_host:remote_port
    pattern = r'-L\s+(\d+):([^:]+):(\d+)'
    tunnels = []
    
    for match in re.finditer(pattern, ssh_cmd):
        tunnels.append({
            'local_port': match.group(1),
            'node': match.group(2),
            'remote_port': match.group(3)
        })
    
    # Extract user@host at the end
    user_host_match = re.search(r'(\S+)@(\S+)\s*$', ssh_cmd)
    if user_host_match:
        user = user_host_match.group(1)
        host = user_host_match.group(2)
    else:
        user = "user"
        host = "host"
    
    return tunnels, user, host


def generate_powershell(tunnels: list[dict], user: str, host: str) -> str:
    """Generate PowerShell script content."""
    
    lines = [
        "# Auto-generated SSH tunnel script",
        "# Each tunnel runs in a separate window to avoid file descriptor limits",
        "",
        f'$user = "{user}"',
        f'$host_addr = "{host}"',
        "",
        "$tunnels = @(",
    ]
    
    for i, t in enumerate(tunnels):
        comma = "," if i < len(tunnels) - 1 else ""
        lines.append(f'    @{{LocalPort = {t["local_port"]}; Node = "{t["node"]}"; RemotePort = {t["remote_port"]}}}{comma}')
    
    lines.append(")")
    lines.append("")
    lines.append("""foreach ($t in $tunnels) {
    $lp = $t.LocalPort
    $node = $t.Node
    $rp = $t.RemotePort
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Tunnel: localhost:$lp -> ${node}:$rp' -ForegroundColor Green; ssh -N -L ${lp}:${node}:${rp} ${user}@${host_addr}"
    
    Write-Host "Started: localhost:$lp -> ${node}:$rp" -ForegroundColor Cyan
    Start-Sleep -Milliseconds 500
}

Write-Host "`nAll $($tunnels.Count) tunnels started!" -ForegroundColor Green
Write-Host "Use -ns $($tunnels.Count) with your Python script" -ForegroundColor Yellow""")
    
    return "\n".join(lines)


def main():
    # Check for --run flag
    auto_run = "--run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--run"]
    
    # Get SSH command from argument or stdin
    if args:
        ssh_cmd = " ".join(args)
    else:
        print("Paste your SSH tunnel command (or press Enter for example):")
        ssh_cmd = input().strip()
        if not ssh_cmd:
            ssh_cmd = "ssh -L 8000:node1:8000 -L 8001:node2:8000 user@login.example.com"
    
    print(f"\nParsing: {ssh_cmd[:80]}...")
    
    tunnels, user, host = parse_ssh_command(ssh_cmd)
    
    if not tunnels:
        print("Error: No -L port forwards found in command!")
        sys.exit(1)
    
    print(f"Found {len(tunnels)} tunnel(s):")
    for t in tunnels:
        print(f"  localhost:{t['local_port']} -> {t['node']}:{t['remote_port']}")
    print(f"User: {user}@{host}")
    
    # Generate PowerShell script
    ps_content = generate_powershell(tunnels, user, host)
    
    # Write to file
    output_path = Path(__file__).parent / "start_tunnels.ps1"
    output_path.write_text(ps_content, encoding='utf-8')
    
    print(f"\nGenerated: {output_path}")
    
    # Auto-run if --run flag is set
    if auto_run:
        import subprocess
        print("\nStarting tunnels...")
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(output_path)])
    else:
        print(f"Run with:  .\\leonardo_scripts\\start_tunnels.ps1")
        print(f"Or re-run with --run flag to auto-start")


if __name__ == "__main__":
    main()
