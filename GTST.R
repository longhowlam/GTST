library(mxnet)
library(readr)


##########  model parameters ###############
batch.size         = 64
seq.len            = 64
num.hidden         = 64
num.embed          = 16
num.lstm.layer     = 4
num.round          = 64
learning.rate      = 0.1
wd                 = 0.00001
clip_gradient      = 1
update.period      = 1
train.val.fraction = 0.8

### #####



###### auxilary functions ####################

make.dict = function(text, max.vocab=10000) {
  text = strsplit(text, '')
  dic  = list()
  idx  = 1
  for (c in text[[1]]) {
    if (!(c %in% names(dic))) {
      dic[[c]] = idx
      idx = idx + 1
    }
  }
  if (length(dic) == max.vocab - 1)
    dic[["UNKNOWN"]] = idx
  cat(paste0("Total unique char: ", length(dic), "\n"))
  return (dic)
}

make.data = function(file.path, seq.len=32, max.vocab=10000, dic=NULL) {
  fi   = file(file.path, "r")
  text = paste(readLines(fi), collapse="\n")
  close(fi)
  
  if (is.null(dic))
    dic = make.dict(text, max.vocab)
  lookup.table = list()
  for (c in names(dic)) {
    idx = dic[[c]]
    lookup.table[[idx]] = c 
  }
  
  char.lst = strsplit(text, '')[[1]]
  num.seq  = as.integer(length(char.lst) / seq.len)
  char.lst = char.lst[1:(num.seq * seq.len)]
  data     = array(0, dim=c(seq.len, num.seq))
  idx      = 1
  for (i in 1:num.seq) {
    for (j in 1:seq.len) {
      if (char.lst[idx] %in% names(dic))
        data[j, i] <- dic[[ char.lst[idx] ]]-1
      else {
        data[j, i] <- dic[["UNKNOWN"]]-1
      }
      idx <- idx + 1
    }
  }
  return (list( data = data, dic = dic, lookup.table = lookup.table))
}

drop.tail = function(X, batch.size) {
  shape = dim(X)
  nstep = as.integer(shape[2] / batch.size)
  return (X[, 1:(nstep * batch.size)])
}


get.label = function(X) {
  label = array(0, dim=dim(X))
  d = dim(X)[1]
  w = dim(X)[2]
  for (i in 0:(w-1)) {
    for (j in 1:d) {
      label[i*d+j] <- X[(i*d+j)%%(w*d)+1]
    }
  }
  return (label)
}

### ####



############### import GTST samenvattingen from csv and create input data for RNN/LSTM #############

gtst_daily_data = read_csv("data/GTST_Daily_data.csv", col_types = cols(datums = col_skip(),  datums2 = col_skip()))
write.table(gtst_daily_data, "data/input2.txt", quote = FALSE, col.names = FALSE, row.names = FALSE )




############## create input arrays from input data data ################## 

ret = make.data("data/input2.txt", seq.len=seq.len) #GTSTS

ret = make.data("data/50shades.txt", seq.len=seq.len) ## 50 shades grey



X             = ret$data
dic           = ret$dic
lookup.table  = ret$lookup.table
vocab         = length(dic)

shape         = dim(X)
size          = shape[2]

X.train.data  = X[, 1:as.integer(size * train.val.fraction)]
X.val.data    = X[, -(1:as.integer(size * train.val.fraction))]
X.train.data  = drop.tail(X.train.data, batch.size)
X.val.data    = drop.tail(X.val.data, batch.size)

X.train.label = get.label(X.train.data)
X.val.label   = get.label(X.val.data)

X.train       = list(data = X.train.data, label = X.train.label)
X.val         = list(data = X.val.data, label = X.val.label)

dim(X.train.data)
### ####

############ Training LSTM Model ##############################

tic <- proc.time()
model <- mx.lstm(
  X.train, X.val, 
  ctx = mx.cpu(),
  num.round = num.round, 
  update.period = update.period,
  num.lstm.layer = num.lstm.layer, 
  seq.len = seq.len,
  num.hidden = num.hidden, 
  num.embed = num.embed, 
  num.label = vocab,
  batch.size = batch.size, 
  input.size = vocab,
  initializer = mx.init.uniform(0.1), 
  learning.rate = learning.rate,
  wd = wd,
  clip_gradient = clip_gradient
)

tic2 = proc.time() - tic

infer.model = mx.lstm.inference(
  num.lstm.layer = num.lstm.layer,
  input.size     = vocab,
  num.hidden     = num.hidden,
  num.embed      = num.embed,
  num.label      = vocab,
  arg.params     = model$arg.params,
  ctx            = mx.cpu()
)
### ####


######### generate text from model ####################################

start = 'a'
seq.len = 2175
random.sample = TRUE

last.id = dic[[start]]
out = "a"
for (i in (1:(seq.len-1))) {
  input       = c(last.id-1)
  ret         = mx.lstm.forward(infer.model, input, FALSE)
  infer.model = ret$model
  prob        = ret$prob
  last.id     = make.output(prob, random.sample)
  out         = paste0(out, lookup.table[[last.id]])
}

cat (paste0(out, "\n"))
write.table(out, "50_SHADES_NEW.txt", col.names=FALSE, row.names = FALSE)




