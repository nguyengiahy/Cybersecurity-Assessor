# NLP PIPELINE

This folder contains the implementation of all NLP Processes for ACSIMA prototype.
**When running the NLP pipeline, please make sure you allocate at least 12GB of RAM to Docker (each criteria takes around 2GB) and as many CPU cores as you have available.**

## Folder structure

    .
    ├── criteria                    # The complete implementation of NLP pipelines, each to assess one criteria
    ├── models                      # AI Trained Models ready to be used for NLP pipelines
    ├── preprocessors               # NLP Tools to process the input data before feeding into the NLP models
    ├── parameters.py               # List of parameters to tweak for each NLP pipeline
    ├── pipeline.py
    └── README.md

## Models
- [BERT](./models/bert/README.md)

## Parameter tweaking

`parameters.py` contains the list of paramaters that need to be modified to gear the NLP models towards generating the expected outcomes.

### Global parameters

Global parameters are used for all criteria.

#### Custom Sentence tokenizer

The sentence tokenizer is responsible for extracting the sentence from a piece of text. The sentence tokenizer is based on the Spacy trained model. Spacy will by default determine some punctuation marks / keywords to be the end of a sentence. In addition to the Spacy's original sentence-extraction rules, you can specify your own sentence boundaries by modifying the `SENTENCE_BOUNDARY_KEYWORDS`. If your models do not expect Spacy's default marks / keywords to be the sentence boundaries, you can also add those marks / keywords into `SENTENCE_NONE_BOUNDARY_KEYWORDS`.

### Criteria-specific parameters

Criteria-specific parameters are used for a single criteria only.

For example, `C9` requires extracting the relevant keywords for identifing `PII` information. To tweak the keyword extractor, add / remove / modify the `C9_KEYWORDS`.

## Criteria Pipeline Class Implementation
Each criteria is implemented as its own class. The main `pipeline.py` file contains all criteria pipelines and preprocesses privacy policy texts before sending them to each criteria. Each criteria class has a single BERT model that it evaluates. Furthermore, specific evaluation of metadata can also be implemented on a per-criteria basis. It should also be noted that the criteria classes have been upgraded to use python multiprocessing. This enables a more parallel assessment of multiple criteria, especially as more are added. However, RAM usage is increased due to the overhead of new processes and the need for each process to load the `bert_raw` model that could otherwise be shared. Overall, the use of multiprocessing results in each new criteria process added taking around 1.5GB - 2GB of RAM, depending on the size of the trained model for that criteria.

To get the most benefit from the multiprocessing it is recommended to allocate as many CPU cores as possible to Docker. Each criteria acts as a new system process so having at least this many cores allocated, plus one for the main process, is desired. However, additional allocation beyond this can also help too.

### How to add another pipeline class
- Step 1: Ensure you have your trained model for your criteria exported to either DockerHub or in the saved_models local directory. See [BERT training](./models/bert/README.md) for more details
- Step 2: Create a new file under `criteria/` called `CXpipeline.py` with `X` being your criteria number
- Step 3: Add your keywords array for finding relevant paragraphs to `parameters.py`
- Step 4: In the same `parameters.py`, update the `NUM_CRITERIA` value to represent the total number of criteria implemented
- Step 5: Back in the new criteria pipeline class, you can copy the `__init__` method from other pipeline class since they will mostly be the same. You will need to update the `CRITERIA_PATH` variable to be the path to your exported model from Step 1 as well as updating the keywords import line
- Step 6: Implement your `run_assessment` method, currently they are mostly the same, but for some criteria, preprocessing might be different, particularly if other metadata is also considered
- Step 7: Implement the `run` and `get_keywords` methods with minor name changes for the specific criteria keywords and print messages
- Step 8: Go to `pipeline.py` file, within the `instantiate_criteria_queues` method, add this line `self.cXinput_queue = Queue(INPUT_QUEUE_SIZE)` with `X` being your criteria number
- Step 9: In the same file, go to the `instantiate_criteria_pipelines` method, add this line `self.cXpipeline = CXpipeline(self.cXinput_queue, self.assessment_begin_event, self.assessment_queue)` with `X` being your criteria number (you will also need to add an import statement for the new criteria class)
- Step 10: Within the `__init__` method of the same file, look for `self.criteria_pipeline` and add another item within that dictionary as `"cX": { "pipeline": self.cXpipeline, "input": self.cXinput_queue }` with `X` being your criteria number
- Step 11: Consider the need to allocate more RAM to Docker, depending on whether the usage was already previously close to the capacity