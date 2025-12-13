#!/bin/bash
# Start training script

cd /home/felipeEngin/Documents/deepfake_image_video

echo "Starting training..."
echo ""


# Start training in screen session
screen -S deep bash -c "cd /home/felipeEngin/Documents/deepfake_image_video && python train.py --config config.yaml 2>&1 | tee training_output.log"

echo ""
echo "Training started in screen session 'deep'"
echo ""
echo "To attach: screen -r deep"
echo "To detach: Press Ctrl+A then D"
echo "To view logs: tail -f training_output.log"




