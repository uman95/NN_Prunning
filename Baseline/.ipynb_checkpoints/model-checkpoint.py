{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mobilenet import MobileNet\n",
    "from vgg import VGG\n",
    "from resnet import ResNet18, ResNet50\n",
    "from config import cfg\n",
    "\n",
    "def model():\n",
    "    opt = cfg\n",
    "    if opt.model == 'MobileNet':\n",
    "        return MobileNet\n",
    "    elif opt.model == 'ResNet18':\n",
    "        return ResNet18\n",
    "    elif opt.model == 'ResNet50':\n",
    "        return ResNet50\n",
    "    else:\n",
    "        return VGG"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
