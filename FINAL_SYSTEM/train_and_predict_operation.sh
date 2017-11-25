# the keras model is trained using tensorflow as backend
# the question should be enclosed with in quotes
# sample example run
# bash train_and_predict_operation.sh ../DATASETS/train_memnn_19012017 ../DATASETS/test_memnn_19012017 parm-test model.json model-wts question_to_answer
# run in python3
#!/usr/bin/env bash
training_file=$1
test_file=$2
parameters_file=$3
model_json_file=$4
model_weights_file=$5
question_file=$6
python3 train_model_for_operation_prediction.py --train $training_file --test $test_file --parm $parameters_file --json $model_json_file --weights $model_weights_file
python3 predict_operation_and_operands_to_find_answer.py --json $model_json_file --weight $model_weights_file --parm $parameters_file --ques $question_file