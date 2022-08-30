# PTMAPIRec
Leveraging Pre-trained Models of Source Code in API Recommendation.

## Data Processing

### Mongo Data Format

There are 4 collections in MongoDB.

`api` example:

```json
{
    "signature" : "android.os.Bundle.putString(java.lang.String, java.lang.String)",
    "inParams" : [
        "java.lang.String",
        "java.lang.String"
    ],
    "outputParams" : "void",
    "className" : "android.os.Bundle",
    "apiName" : "putString",
    "type" : "android"
}
```

`project` example:

```json
{
    "filePath" : "FILE PATH",
    "projectName" : "PROJECT NAME",
    "type" : "android",
    "split" : "test-long",
}
```

`class` example:

```json
{
    "source" : "SOURCE CODE",
    "fileName" : "FILE NAME.java",
    "projectEntity" : DBRef("project", ObjectId("..."))
}
```


`method` example:

```json
{
    "source" : "SOURCE CODE",
    "maskedSource" : "SOURCE CODE with [HOLE]}",
    "hasAPIs" : true,
    "apiEntityList" : [
        DBRef("api", ObjectId("...")),
        DBRef("api", ObjectId("..."))
    ],
    "classEntity" : DBRef("class", ObjectId("...")),
}
```

### Processing

Set `MONGO_SERVER` in [data_prepare/mongo_service.py](data_prepare/mongo_service.py) first, and then process MongoDB data using [data_prepare/prepare_mongo_data.py](data_prepare/prepare_mongo_data.py). 

### Data Source

* API Bench: https://github.com/JohnnyPeng18/APIBench
* CodeBERT: https://huggingface.co/microsoft/codebert-base-mlm
* CodeGPT: https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2
* CodeT5: https://huggingface.co/Salesforce/codet5-base

## Train

Example:

```bash

python main.py --train \
    --model_type gpt \
    --pre_trained_tokenizer pretrained_model/CodeGPT-small-java-adaptedGPT2 \
    --pre_trained_model pretrained_model/CodeGPT-small-java-adaptedGPT2 \
    --vocab_file data/api_vocab/android_api.json \
    --train_dataset data/dataset/android/train.json \
    --model_sig ptmapirec \
    --max_seq_len 512 \
    --ignore_pt
```

## Test

Example:

```bash
python main.py --train \
    --model_type gpt \
    --pre_trained_tokenizer pretrained_model/CodeGPT-small-java-adaptedGPT2 \
    --pre_trained_model pretrained_model/CodeGPT-small-java-adaptedGPT2 \
    --vocab_file data/api_vocab/android_api.json \
    --test_dataset data/dataset/android \
    --model_sig ptmapirec \
    --max_seq_len 512 \
    --saved_model my_model_checkpoint.pt
    --ignore_pt

```
