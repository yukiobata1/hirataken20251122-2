# Running EPSR in Background with nohup

This directory contains helper scripts to run the EPSR simulation in the background, allowing you to disconnect from SSH without stopping the simulation.

## Quick Start

### 1. Start EPSR in background
```bash
./run_epsr_nohup.sh
```

This will:
- Start the EPSR simulation in the background
- Create a log file in `logs/epsr_YYYYMMDD_HHMMSS.log`
- Save the process ID to `logs/epsr.pid`
- Print instructions for monitoring

### 2. Check status
```bash
./epsr_status.sh
```

This will show:
- Whether EPSR is running
- The current PID
- The last 20 lines of the log file

### 3. Monitor in real-time
```bash
tail -f logs/epsr_YYYYMMDD_HHMMSS.log
```

Or check the most recent log:
```bash
tail -f $(ls -t logs/epsr_*.log | head -1)
```

Press `Ctrl+C` to stop following the log (this won't stop the simulation).

### 4. Stop EPSR
```bash
./epsr_stop.sh
```

This will gracefully stop the EPSR simulation.

## Example Session

```bash
# Start simulation
$ ./run_epsr_nohup.sh
Starting EPSR simulation in background...
Log file: logs/epsr_20251129_123456.log
PID file: logs/epsr.pid
Process started with PID: 12345

To monitor progress:
  tail -f logs/epsr_20251129_123456.log

# You can now disconnect from SSH
$ exit

# ... later, reconnect and check status ...
$ ./epsr_status.sh
EPSR is running (PID: 12345)
Log file: logs/epsr_20251129_123456.log

=== Last 20 lines of log ===
[EPSR output]

# Stop when done
$ ./epsr_stop.sh
Stopping EPSR (PID: 12345)...
EPSR stopped.
```

## Direct Python Execution

If you prefer to run directly with nohup:

```bash
nohup python scripts/main_epsr.py > logs/epsr.log 2>&1 &
echo $! > logs/epsr.pid
```

## Troubleshooting

### Check if process is still running
```bash
ps -p $(cat logs/epsr.pid)
```

### Find all EPSR processes
```bash
ps aux | grep main_epsr.py
```

### Kill manually if scripts fail
```bash
kill $(cat logs/epsr.pid)
# or force kill
kill -9 $(cat logs/epsr.pid)
```

### View all log files
```bash
ls -lht logs/
```
