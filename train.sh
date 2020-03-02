echo "Running ResNet18 on GreyScale Cifar10"
python main.py --model=='resnet18' --nEpochs=100

echo "Running ResNet50 on GreyScale Cifar10"
python main.py --model=='resnet50' --nEpochs=100

# Add your command line interface to run your model.
