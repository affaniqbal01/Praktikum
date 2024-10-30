import optuna
import mlflow
import yaml
from ultralytics import YOLO
from ultralytics import settings
import re
import numpy as np

# Update a setting
settings.update({'mlflow': False})



def get_fourth_value_as_float(results_dict):
    # Extract the fourth value from the dictionary
    fourth_value = list(results_dict.values())[3]
    # Convert to float (if not already a float)
    fourth_value_float = float(fourth_value)
    return fourth_value_float



def objective(trial):
    # Suggest hyperparameters
    epoch = trial.suggest_int('epochs', 150, 300)  # Example range for epochs
    batch = trial.suggest_int('batch', 16, 32) 
    # batch = trial.suggest_categorical('batch', [16, 32])  # Restrict to only 18 or 32
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-2, log=True)  # Log scale for learning rate



    # Load the configuration file
    with open(r"params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow.end_run()
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    run_name = params['name']
    experiment_name = params['experiment_name']

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        mlflow.create_experiment(experiment_name, "mlflow-artifacts:/0")
        print("Experiment created")
    else:
        print("Experiment with the same name already exists")
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Load a pre-trained model
            model = YOLO(params['model_type'])
            PROJECT = "my_proj"

            # Train
            results = model.train(
                        data='data.yaml',
                        # imgsz=params['imgsz'],
                        batch=batch,
                        epochs=epoch,  # Use the suggested epoch
                        # optimizer=params['optimizer'],
                        lr0=lr0,
                        patience = params['patience'],
                        # seed=params['seed'],
                        # hsv_s=params['hsv_s'],
                        # # hsv_v=params['hsv_v'],
                        # # scale=params['scale'],
                        pretrained=params['pretrained'],    
                        project=PROJECT,
            )

            run, active_run = mlflow, mlflow.active_run()
            print("active run id:", active_run.info.run_id)


            # Log best fit and last fit in mlflow
            mlflow.log_artifact(model.trainer.last)
            mlflow.log_artifact(model.trainer.best)

            # Log params in mlflow
            run.log_params(vars(model.trainer.model.args))
            
            # Log metrics in mlflow
            metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in model.trainer.metrics.items()}
            run.log_metrics(metrics=metrics_dict, step=model.trainer.epoch)

            # Log the PyTorch model to the artifact location specified by 'artifact_path'
            mlflow.pyfunc.log_model(artifact_path="model", artifacts={'model_path': str(model.trainer.save_dir)}, python_model=mlflow.pyfunc.PythonModel())

            # trying to get the matrics(not a single value)    
            # val_results = model.val(data = 'data.yaml')
            # mAP = np.array(val_results.curves_results[0]) 
            # list_of_floats = mAP.tolist()

            mAP_50_95_FLOAT = get_fourth_value_as_float(results.results_dict)


        except Exception as e:
            print(f"An error occurred during training: {e}")
            return None  # Return None or a specific value in case of failure

    return mAP_50_95_FLOAT  # Return mAP50-95 for optimization
    # return list_of_floats  # Return mAP50-95 for optimization

# Create and optimize the study
study = optuna.create_study(storage='sqlite:///new_study.db', direction='maximize')
study.optimize(objective, n_trials=6)

# Check if there are completed trials
if len(study.trials) == 0 or all(trial.value is None for trial in study.trials):
    print("No successful trials were completed.")
else:
    # Print the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
