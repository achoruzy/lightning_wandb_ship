#!/bin/bash

wandb sweep sweep.yaml

wandb agent industark/Ships_wandb_course/xxxxxxx --count 20