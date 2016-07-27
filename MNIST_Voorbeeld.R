library(mxnet)
library(readr)


#### inlezen MNIST data en preparatie voor mxnet ####

MNIST_DIGITS = read_csv('data/train.csv')

### split train data in train en test
N = dim(MNIST_DIGITS)[1]

tr = sample(1:N, size = floor(0.75*N))

train = data.matrix(MNIST_DIGITS[tr,])
test  = data.matrix(MNIST_DIGITS[-tr,])

train.x = train[,-1]
train.y = train[,1]

train.x <- t(train.x/255)
test.x = t(test[,-1]/255)
test.y = test[,1]

table(train.y)
table(test.y)


######################### VANILLA ####################################################
####
#### traditioneel fully connected netwerk specificeren

data = mx.symbol.Variable("data")
fc1  = mx.symbol.FullyConnected(data, name="fc1", num_hidden = 256)
act1 = mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2  = mx.symbol.FullyConnected(act1, name="fc2", num_hidden = 128)
act2 = mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3  = mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)

softmax = mx.symbol.SoftmaxOutput(fc3, name="sm")

######  trainen van model
time1 = proc.time()

devices <- mx.cpu()

mx.set.seed(0)
modelMNIST <- mx.model.FeedForward.create(
  softmax, X=train.x, y=train.y,
  ctx=devices, num.round=10, array.batch.size=100,
  learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
  initializer=mx.init.uniform(0.07),
  epoch.end.callback=mx.callback.log.train.metric(100)
)

time2 = proc.time() - time1

###  pas model toe op test set
preds = predict(modelMNIST, test.x)
preds = data.frame(t(preds), label = test.y)
preds$prediction = apply(preds[,1:10], 1,which.max) - 1
preds$fout = ifelse(preds$prediction != preds$label, 1,0)

mean(preds$fout)
# 0.02504762
1-mean(preds$fout)
# 0.9749524

############ LE NET ##################################################################
###
### LE NET convolutional netwerk architectuur

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

### input data moet hervormd worden tot arrays
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))


##### train model
device.cpu <- mx.cpu()

mx.set.seed(0)
tic <- proc.time()
modelLENET <- mx.model.FeedForward.create(
  lenet, X=train.array, y=train.y,
  ctx=device.cpu, num.round=10, array.batch.size=150,
  learning.rate=0.05, momentum=0.9, wd=0.00001,
  eval.metric=mx.metric.accuracy,
  epoch.end.callback=mx.callback.log.train.metric(100)
)
tic2 = proc.time()-tic


predL = predict(modelLENET, test.array)
predL = data.frame(t(predL), label = test.y)
predL$prediction = apply(predL[,1:10], 1,which.max) - 1
predL$fout = ifelse(predL$prediction != predL$label, 1,0)

mean(predL$fout)
#  0.0172381
1-mean(predL$fout)
# 0.9827619


