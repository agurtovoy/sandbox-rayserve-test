Developing on EC2's Deep Learning AMI instance:

# Installing

- `source activate pytorch_latest_p37`
- `pip install "ray[serve]"`
- `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html`
- `pip install opencv-python`

# Running

- `ray start --head`
- `python serve_model.py`

# Updating the model

- `python serve_model.py -u`
