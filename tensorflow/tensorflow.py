import tensorflow as tf #TensorFlowfmf tf라는 변수로 저장
from tensorflow.examples.tutorials.mnist import input_data #tensorflow의 예제들에서 input_data를 불러옴

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True) # mnist에 Dataset을 불러옴

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # x값을 부정소수점으로 이루어진 2차원 텐서로 표현
W = tf.Variable(tf.zeros([784, 10])) # 편향값
b = tf.Variable(tf.zeros([10])) # 가중치
y = tf.nn.softmax(tf.matmul(x, W) + b) # 계산값

y_ = tf.placeholder(tf.float32, [None, 10]) # 새 플레이스홀더를 추가 - 교차엔트로피 구현위함

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 교차엔트로피 구현 - 모든 이미지들에 대한 교차엔트로피의 합
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 교차엔트로피의 최소화

# Session
init = tf.initialize_all_variables() # 변수 초기화

sess = tf.Session() # 세션에서 모델시작
sess.run(init) # 변수 초기화

# Learning
for i in range(1000): # 1000번 실행 - 1000번 학습시킴 
  batch_xs, batch_ys = mnist.train.next_batch(100) # 100개의 무작위 데이터들을 가져옴
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 세션에서 모델 시작

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 정확도 계산 (True = 1, False = 0)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 얼마나 맞았는지 평균값을 계산 

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 테스트 데이터의 정확도 확인