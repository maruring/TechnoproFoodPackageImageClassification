# 予測だけをしたいときに使用する
from get_original_model import MyEnsembleModel
from prepare_training_elements import PrepareTrainingElements

if __name__ == "__main__":
    model_getter = PrepareTrainingElements()