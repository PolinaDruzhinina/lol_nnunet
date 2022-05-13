## Build docker and run inference

1. If you want to build docker yourself, perhaps with your test file, you need
```python
docker build --tag my_image - < Dockerfile 
```

2. To run ours inference:

```python
docker run --rm -it -v path_to_input/:/input -v /path_to_otput/:/output -v path_to_model_weights/epoch\=25-val_dice.ckpt:/home/epoch\=25-val_dice.ckpt inferene:latest --resume_from_checkpoint /home/epoch\=25-val_dice.ckpt --gpus 1 --num_encoding_blocks 5
```

If you wish, you can change the parameters of the model, GPU device, weights, etc., or use the standard ones.