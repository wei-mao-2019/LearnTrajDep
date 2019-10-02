## Learning Trajectory Dependencies for Human Motion Prediction
This is the code for the paper

Wei Mao, Miaomiao Liu, Mathieu Salzmann, Hongdong Li. 
[_Learning Trajectory Dependencies for Human Motion Prediction_](https://arxiv.org/abs/1908.05436). In ICCV 19.

### Dependencies

* cuda 9.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.3.1.
* [progress 1.5](https://pypi.org/project/progress/)

### Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

### Quick demo and visualization

For a quick demo, you can train for a few epochs and visualize the outputs
of your model.

To train, run
```bash
python main.py --epoch 5 --input_n 10 --output 10 --dct_n 20 --data_dir [Path To Your H36M data]/h3.6m/dataset/
```

Visualize the results of pretrained model for predictions on angle space on H36M dataset.
* change the model path
* then run the command below
```bash
python demo.py --input_n 10 --output_n 10 --dct_n 20 --data_dir [Path To Your H36M data]/h3.6m/dataset/
```
### Training commands
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train on angle space,
```bash
python main.py --data_dir "[Path To Your H36M data]/h3.6m/dataset/" --input_n 10 --output_n 10 --dct_n 20 --exp [where to save the log file]
```
```bash
python main_cmu.py --data_dir_cmu "[Path To Your CMU data]/cmu_mocap/" --input_n 10 --output_n 25 --dct_n 35 --exp [where to save the log file]
```
```bash
python main_3dpw.py --data_dir_3dpw "[Path To Your 3DPW data]/3DPW/sequenceFiles/" --input_n 10 --output_n 30 --dct_n 40 --exp [where to save the log file]
```
To train on 3D space,
```bash
python3 main_3d.py --data_dir "[Path To Your H36M data]/h3.6m/dataset/" --input_n 10 --output_n 10 --dct_n 15 --exp [where to save the log file]
```
```bash
python main_cmu_3d.py --data_dir_cmu "[Path To Your CMU data]/cmu_mocap/" --input_n 10 --output_n 25 --dct_n 30 --exp [where to save the log file]
```
```bash
python main_3dpw_3d.py --data_dir_3dpw "[Path To Your 3DPW data]/3DPW/sequenceFiles/" --input_n 10 --output_n 30 --dct_n 35 --exp [where to save the log file]
```


### Results
We re-run our code 2 more times under different setups and the overall average results at different time are reported below.

* Human3.6-short-term prediction on angle space (top) and 3D coordinate (bottom)

|                | 80ms   | 160ms  | 320ms  | 400ms  |
|----------------|------|------|------|------|
| pre-trained | 0.27 | 0.51 | 0.83 | 0.95 |
| test_run_1     | 0.28 | 0.52 | 0.84 | 0.96 |
| test_run_2     | 0.28 | 0.52 | 0.84 | 0.96 |
|----------------|------|------|------|------|
| pre-trained | 12.1 | 25.0 | 51.0 | 61.3 |
| test_run_1 | 12.1 | 24.6 | 50.4 | 61.1 |
| test_run_2 | 12.1 | 24.8 | 50.5 | 61.2 |

* Human3.6-long-term prediction

|             | 560ms  |1000ms|
|-------------|--------|------|
| pre-trained | 0.90   | 1.27 |
| test_run_1  | 0.91   | 1.25 |
| test_run_2  | 0.92   | 1.27 |
|-------------|--------|------|
| pre-trained | 50.4   | 71.0 |
| test_run_1  | 51.2   | 71.6 |
| test_run_2  | 51.6   | 70.9 |


* CMU-mocap

|             | 80ms | 160ms | 320ms | 400ms | 1000ms |
|-------------|------|-------|-------|-------|--------|
| pre-trained | 0.25 | 0.39  | 0.68  | 0.79  | 1.33   |
| test_run_1  | 0.26 | 0.41  | 0.72  | 0.84  | 1.35   |
| test_run_2  | 0.26 | 0.41  | 0.71  | 0.83  | 1.38   |
|-------------|------|-------|-------|-------|--------|
| pre-trained | 11.5 | 20.4  | 37.8  | 46.8  | 96.5   |
| test_run_1  | 11.3 | 19.8  | 36.9  | 45.5  | 92.7   |
| test_run_2  | 11.3 | 19.7  | 37.2  | 46.0  | 94.0   |

* 3DPW

|             | 200ms | 400ms | 600ms | 800ms | 1000ms |
|-------------|-------|-------|-------|-------|--------|
| pre-trained | 0.64  | 0.95  | 1.12  | 1.22  | 1.27   |
| test_run_1  | 0.64  | 0.97  | 1.12  | 1.22  | 1.28   |
| test_run_2  | 0.64  | 0.95  | 1.11  | 1.21  | 1.27   |
|-------------|-------|-------|-------|-------|--------|
| pre-trained | 35.6  | 67.8  | 90.6  | 106.9 | 117.8  |
| test_run_1  | 36.7  | 69.6  | 90.8  | 105.0 | 115.3  |
| test_run_2  | 35.8  | 69.1  | 93.2  | 110.9 | 121.7  |


### Citing

If you use our code, please cite our work

```
@inproceedings{wei2019motion,
  title={Learning Trajectory Dependencies for Human Motion Prediction},
  author={Wei, Mao and Miaomiao, Liu and Mathieu, Salzemann and Hongdong, Li},
  booktitle={ICCV},
  year={2019}
}
```

### Acknowledgments

Some of our evaluation code and data process code was adapted/ported from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). The overall code framework (dataloading, training, testing etc.) is adapted from [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). 

### Licence
MIT
