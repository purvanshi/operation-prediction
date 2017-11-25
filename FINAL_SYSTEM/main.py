import train_model_for_operation_prediction
import predict_operation_and_operands_to_find_answer

train ="train.txt"
testfile ="test.txt"
para ="parameters_GRU_21112017_tensorflow_pre_padding_0.5dropout"
model ="model_json_GRU_21112017_tensorflow_pre_padding_0.5dropout.json"
weights = "weights_GRU_21112017_tensorflow_pre_padding_0.5dropout"
question = "question_to_answer.txt"
train_model_for_operation_prediction.main(train,testfile,model,weights)
predict_operation_and_operands_to_find_answer.main(model,weights,para,question)