
import azureml.core
from azureml.core import (ComputeTarget,Environment, Experiment, Model,
                          ScriptRunConfig, Workspace)
from azureml.core.conda_dependencies import CondaDependencies

def main():
    # Create a Python environment for the experiment
    py_torch = Environment("pytorch-env")

    training_folder = '/Users/georgioszefkilis/mlops_fork/dtu_mlops/07_distributed_training/'
    ws = Workspace.from_config()
    print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

     # Set the compute target
    compute_target = ComputeTarget(ws, 'MlopsParallel')
    print('Ready to use compute target: {}'.format(compute_target.name))

    # Ensure the required packages are installed (we need pip, scikit-learn and Azure ML defaults)
    packages = CondaDependencies.create(conda_packages=['pip', 'torvh','torchvision'],
                                    pip_packages=['azureml-defaults'])
    py_torch.python.conda_dependencies = packages

    # Create a script config
    script_config = ScriptRunConfig(source_directory=training_folder,
                                script='cloud_training.py',
                                environment=py_torch,
                                compute_target=compute_target) 

    experiment = Experiment(workspace=ws, name='fashion-classifier-training-test')
    run = experiment.submit(config=script_config)

    # Get logged metrics and files
    print('Getting run metrics')
    metrics = run.get_metrics()
    for key in metrics.keys():
        print(key, metrics.get(key))
    print('\n')

    # Register the model
  
    run.register_model( model_path = 'outputs/fashion_trainer.pkl',
                        model_name='fashion_trainer',
                        tags={'Training context':'Script'})

    # List registered models
    for model in Model.list(ws):
        print(model.name, 'version:', model.version)
        for tag_name in model.tags:
            tag = model.tags[tag_name]
            print ('\t',tag_name, ':', tag)
        for prop_name in model.properties:
            prop = model.properties[prop_name]
            print ('\t',prop_name, ':', prop)
        print('\n') 


if __name__ == '__main__':
    main()                             