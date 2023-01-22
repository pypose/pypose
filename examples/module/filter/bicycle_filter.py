import argparse
from bicycle import Bicycle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filer Model Example')
    parser.add_argument("--model_name", type=str, default='ukf', help="ekf or ukf")
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--weight_method", type=str, default='norm',
                        help="weight method for ukf")
    args = parser.parse_args()

    model_name = args.model_name
    device = args.device
    weight_method = args.weight_method
    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation
    bicycle = Bicycle(T, N, M, q, r, p, model_name, device)
    bicycle.run_estimate(bicycle, weight_method)
