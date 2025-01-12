from train import train_main
from inference import inference
from utils import set_seed
from config import RANDOM_STATE

if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    preprocessors, optimal_thresholds = train_main()
    inference(preprocessors, optimal_thresholds)
