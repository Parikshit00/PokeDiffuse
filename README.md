# Draw a Pokemon and Guess the Pokemon
This is an ongoing side little fun project.

If you are willing to collaborate just open an issue, I will respond.

The project is inspired from Sketch-a-Sketch paper. See the details [here](https://vsanimator.github.io/sketchasketch/).


## Generate Partial Sketch 

To generate partial sketch, see the sketch-a-sketch paper approach. It utilizes holistic edge detection (HED). The implementation of HED is taken from [here](https://github.com/sniklaus/pytorch-hed).

![example HED](./static/example.png)

Implementation of HED on a sample image of an Abra:  
![my_pipeline](./static/data.png)


## Tasks

- [x] Setup data generation pipeline 
- [x] Test on stable diffusion2 sketch model version 
- [ ] Annotate text description for each RGB image
- [ ] Finalize diffusion architecture
- [ ] Finetune architecture models with stable diffusion 2.

