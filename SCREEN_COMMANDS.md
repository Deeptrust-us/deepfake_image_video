# Screen Session Commands

## Training is running in screen session "deep"

### Useful Commands:

**Attach to screen session:**
```bash
screen -r deep
```

**Detach from screen (while inside):**
- Press: `Ctrl+A` then `D`

**List all screen sessions:**
```bash
screen -ls
```

**Kill screen session:**
```bash
screen -X -S deep quit
```

**View training output without attaching:**
```bash
tail -f /home/felipeEngin/Documents/deepfake_image_video/training_output.log
```

**Check if training is still running:**
```bash
screen -r deep
# Then check the output, press Ctrl+A then D to detach
```

### Quick Status Check:

```bash
# Check screen session
screen -ls

# Check latest training output
tail -20 /home/felipeEngin/Documents/deepfake_image_video/training_output.log

# Check GPU usage (if using CUDA)
nvidia-smi
```

### Monitor Training Progress:

```bash
# Option 1: Attach to screen
screen -r deep

# Option 2: Follow log file
tail -f training_output.log

# Option 3: Use TensorBoard (in another terminal)
tensorboard --logdir logs
```

