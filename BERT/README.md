# BERT MODEL

## Training BERT
**NOTE:** It is recommended that you run the training on Google Collab. Otherwise your computer will be screaming.
**NOTE:** Due to Bitbucket incompetencies in handling huge data files, we have to remove the trained BERT models from the repository. The temporary solution now is to download the trained models which are stored on the team's shared google drive and put them under `saved_models` folder. Then, you can run the assessment as usual.

### Without Docker
- Step 0: `cd NlpServer` and run `pip3 install -r requirements.bert.txt` and then run `pip3 install -r requirements.txt`
- Step 1: `cd /NlpServer/applications/assessment/nlpPipeline/models/bert`
- Step 2: Have your pre-processed data ready and put it under the `./data` folder.
- Step 3: Go to `train.py` file and update line `FILE_PATH = os.path.join(CURRENT_DIR, "data/<your-file-name>.csv")` to use your csv file
- Step 4: Find `epochs` variable and change it to the number you want. This equals the number of training iterations taking place. Each runs for about 30-50mins so choose wisely (currently default value is 1)
- Step 5: Find the line `model.save('saved_models/<your-criteria>')` (currently the last line of the file), and update `your-criteria` (e.g c9, c10, etc)
- Step 6: Run `python3 train.py`
- Step 7: Wait for the training to be done. Inside folder `saved_models`, there will be a new subfolders with the criteria number. That contains the exported, trained BERT model.

### With Docker
- Step 0: `docker-compose up --build`
- Step 1: Run `docker ps` and copy the CONTAINER ID of the nlp server
- Step 2: Run `docker exec -it <container-id> bash`
- Step 3: You are now inside the NLP container. You can `cd /applications/assessment/nlpPipeline/models/bert` now
- Step 4: Have your pre-processed data ready and put it under the `./data` folder.
- Step 5: Go to `train.py` file and update line `FILE_PATH = os.path.join(CURRENT_DIR, "data/<your-file-name>.csv")` to use your csv file
- Step 6: Find `epochs` variable and change it to the number you want. This equals the number of training iterations taking place. Each runs for about 30-50mins so choose wisely (currently default value is 1)
- Step 7: Find the line `model.save('saved_models/<your-criteria>')` (currently the last line of the file), and update `your-criteria` (e.g c9, c10, etc)
- Step 8: Run `python3 train.py`
- Step 9: Wait for the training to be done. Inside folder `saved_models`, there will be a new subfolders with the criteria number. That contains the exported, trained BERT model.

### Exported BERT files/folders

- `assets`: This folder is for external files that might be needed for the model, but we don't have anything at the moment.

- `variables`: This holds the learnt weights of the model

- `saved_model.pb`: This is the `MetaGraphDef` - the graph structure of the model, aka the architecture.

- `keras_metadata.pb`: Some metadata for the model. Not sure what's inside.

## Unit testing
- cd `NlpServer/applications/assessment/nlpPipeline/models/bert`
- run `python3 -m unittest`

Note: If testing from inside a running container, it might be necessary to comment out the `self.criteria_pipelines` dictionary keys and the `def instantiate_criteria_pipelines(self):` method in the [pipeline.py class](./../../pipeline.py) to prevent the models from being loaded there. Otherwise, the Docker service might not have enough RAM allocated as running these tests will load the models a 2nd time.

## Containerising trained models

### Using a dedicated custom container and volume
If you have trained a new model for a criteria and want to put that into the model image, follow these steps:
- Step 0: Put your models under `./saved_models`. Don't worry because this folder is included in `.dockerignore` and `.gitignore` so it will not be included in the build/commit.
- Step 1: Go to `Dockerfile.trainedmodels` under project root directory
- Step 2: Add `COPY /NlpServer/applications/assessment/nlpPipeline/models/bert/saved_models/<your-criteria> /acsima/models/<your-criteria>`
- Step 3: Run `./ops/publish_models.sh` from project root directory. Please note you needed to be added as collaborators to this repository before you can push.

We structure the docker file with multiple `COPY`s, each for 1 criteria because we want to utilise docker caching mechanism. All of the previous `COPY`s which have been built and published to Docker Hub will not be pushed again when you run the script. It will only push the new layer (i.e the `COPY` you just did), which saves time.

**Note**: You can pretty much delete your models from `./saved_models` folder after this if you want to save space. But beware that, the next time you want to push a new model, you will need all of the existing models in `./saved_models`, because the Docker instructions are to copy the models from within that folder and build a new image. A solution is still under investigation, but it might be better time-wise if you keep your models locally under that folder for now.

## Using Tensorflow Serving
This option is also about running a separate container to serve the trained models, using the base image of Tensorflow Serving. The container when running will expose REST and gRPC endpoints for other services to communicate with via HTTP requests.

This option is parked on the side at the moment due to timing and resources contraints. But it seems to be impossible because of the limitations on BERT's input format. TL;DR BERT only accepts input in the format of Tensorflow's BatchDataset, which cannot be serialised to be sent via HTTP

However, we still keep these documentations for future reference

Steps to run:
- Run the NLP Server using our script
- `docker pull tensorflow/serving`
- `docker run -p 8501:8501 --name c9 -v "<your-absolute-path-to-c9>:/models/c9/1/" -e MODEL_NAME=c9 -t tensorflow/serving --network=acsima_irs-nlp`
- `docker network connect acsima_irs-nlp c9` - to connect the tensorflow serving to our docker network.

Additional resources for TF Serving:
- [Tensorflow Serving Rest api error](https://stackoverflow.com/questions/62055747/tensorflow-serving-rest-api-error-could-not-find-base-path-models-model-for-se)
- [Tensorflow Serving no versions of servable model found](https://stackoverflow.com/questions/45544928/tensorflow-serving-no-versions-of-servable-model-found-under-base-path)
- [Serving multiple Tensorflow models with Docker](https://stackoverflow.com/questions/53035896/serving-multiple-tensorflow-models-using-docker)
- [Creating your own serving image](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image)
- [Serving ML quickly with Tensorflow Serving](https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008)
- [How to serve ML models with Tensorflow Serving](https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)


### Resources:
- [Save and load model](https://www.tensorflow.org/tutorials/keras/save_and_load)
- [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
- [Save the whole model](https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading)
- [In case when running NLP server and you have out of memory error](https://stackoverflow.com/a/55246936)
- [To increase Docker resource allocation for Windows WSL users](https://docs.microsoft.com/en-us/windows/wsl/wsl-config#configure-global-options-with-wslconfig)
