#/usr/bin/python
# ******************************************************
# Author       : CoffreLv
# Last modified:	2019-04-18 08:24
# Email        : coffrelv@163.com
# Filename     :	diffrent_models.py
# Description  :    声学模型类 
# ******************************************************

'''
修改模型需要对应修改的各个位置：
(1)根据池化层数和每层的pool_size设置：
    <1>get_feature.Data_genetator_all()
        input_Length.append(data_Input.shape[0] // 8 + data_Input.shape[0] %8)  中的8
    <2>get_data_generation.__data_generation()
        input_Length.append(data_Input.shape[0] // 8 + data_Input.shape[0] % 8) 中的8

(2)修改输入维度需要修改：
    <1>acoustic_model.Acoustic_model()
        self.label_max_length
        self.AUDIO_LENGTH
        Reshape层维度
    <2>get_feature.Data_genetator_all()
        audio_length

(3)修改特征维度需要修改
    <1>get_feature.Get_frequecy_feature()
    <2>get_feature.Data_genetator_all()
    <3>汉明窗
    <4>self.AUDIO_FEATURE_LENGTH
    <5>Reshape?

(4)Reshape层维度
    计算方式，n×m = 输入n×m×上一层卷积核数
'''

#一层卷积层
    def Create_model(self): #卷积层*3
        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        #修改音频长度需要对应修改
        layer_f7 = Reshape((200, 3200))(layer_c1)  #Reshape
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_fu9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu9)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

#二层卷积层
    def Create_model(self): #卷积层*3
        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)  #卷积层2
        #修改音频长度需要对应修改
        layer_f7 = Reshape((200, 3200))(layer_c1)  #Reshape
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_fu9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu9)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')

#一层卷积层
    def Create_model(self): #卷积层*3
        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)  #卷积层1
        #修改音频长度需要对应修改
        layer_f7 = Reshape((200, 3200))(layer_c2)  #Reshape
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_fu9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu9)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

    def Create_model(self): #卷积层*3
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_p2 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c1) #池化层2
        layer_p2 = Dropout(0.05)(layer_p2)   #为池化层2添加Dropout
        layer_c3 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p2)    #卷积层3
        layer_c3 = Dropout(0.1)(layer_c3)   #为卷积层3添加Dropout
        layer_p4 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c3) #池化层4
        layer_p4 = Dropout(0.1)(layer_p4)   #为池化层4添加Dropout
        layer_c5 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p4)    #卷积层5
        layer_c5 = Dropout(0.15)(layer_c5)   #为卷积层5添加Dropout
        layer_p6 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c5) #池化层6
        #修改音频长度需要对应修改
        layer_f7 = Reshape((200, 3200))(layer_p6)  #Reshape层7
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_fu9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu9)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

    def Create_model(self): #卷积层*6
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)    #卷积层2
        layer_p3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c2) #池化层3
        layer_p3 = Dropout(0.05)(layer_p3)
        layer_c4 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p3)    #卷积层4
        layer_c4 = Dropout(0.1)(layer_c4)   #为卷积层4添加Dropout
        layer_c5 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c4)    #卷积层5
        layer_p6 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c5) #池化层6
        layer_p6 = Dropout(0.1)(layer_p6)
        layer_c7 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p6)    #卷积层7
        layer_c7 = Dropout(0.15)(layer_c7)   #为卷积层7添加Dropout
        layer_c8 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c7)    #卷积层8
        layer_p9 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c8) #池化层9
        #修改音频长度需要对应修改
        layer_f10 = Reshape((200, 3200))(layer_p9)  #Reshape层10
        layer_f10 = Dropout(0.2)(layer_f10)
        layer_f11 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f10)    #全连接层11
        layer_f11 = Dropout(0.3)(layer_f11)
        layer_fu12 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f11)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu12)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

    def Create_model(self): #卷积层*9
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)    #卷积层2
        layer_c2 = Dropout(0.05)(layer_c2)   #为卷积层2添加Dropout
        layer_c3 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c2)    #卷积层2
        layer_p4 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c3) #池化层4
        layer_p4 = Dropout(0.05)(layer_p4)
        layer_c5 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p4)    #卷积层5
        layer_c5 = Dropout(0.1)(layer_c5)   #为卷积层5添加Dropout
        layer_c6 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c5)    #卷积层6
        layer_c6 = Dropout(0.1)(layer_c6)   #为卷积层6添加Dropout
        layer_c7 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c6)    #卷积层6
        layer_p8 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c7) #池化层8
        layer_p8 = Dropout(0.1)(layer_p8)
        layer_c9 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p8)    #卷积层9
        layer_c9 = Dropout(0.15)(layer_c9)   #为卷积层9添加Dropout
        layer_c10 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c9)    #卷积层10
        layer_c10 = Dropout(0.15)(layer_c10)   #为卷积层10添加Dropout
        layer_c11 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c10)    #卷积层11
        layer_p12 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c11) #池化层12
        #修改音频长度需要对应修改
        layer_f13 = Reshape((200, 3200))(layer_p12)  #Reshape层13
        layer_f13 = Dropout(0.2)(layer_f13)
        layer_f14 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f13)    #全连接层11
        layer_f14 = Dropout(0.3)(layer_f14)
        layer_fu15 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f14)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu15)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

    def Create_model(self): #卷积层*12
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)    #卷积层2
        layer_c3 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c2)  #卷积层3
        layer_c3 = Dropout(0.05)(layer_c3)   #为卷积层3添加Dropout
        layer_c4 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c3)    #卷积层4
        layer_p5 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c4) #池化层5
        layer_p5 = Dropout(0.05)(layer_p5)
        layer_c6 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p5)    #卷积层6
        layer_c6 = Dropout(0.1)(layer_c6)   #为卷积层6添加Dropout
        layer_c7 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c6)    #卷积层7
        layer_c8 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p7)    #卷积层8
        layer_c8 = Dropout(0.1)(layer_c8)   #为卷积层4添加Dropout
        layer_c9 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c8)    #卷积层9
        layer_p10 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c9) #池化层10
        layer_p10 = Dropout(0.1)(layer_p10)
        layer_c11 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p10)    #卷积层11
        layer_c11 = Dropout(0.15)(layer_c11)   #为卷积层11添加Dropout
        layer_c12 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c11)    #卷积层12
        layer_c13 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c12)    #卷积层13
        layer_c13 = Dropout(0.15)(layer_c13)   #为卷积层13添加Dropout
        layer_c14 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c13)    #卷积层14
        layer_p15 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c14) #池化层15
        #修改音频长度需要对应修改
        layer_f16 = Reshape((200, 3200))(layer_p15)  #Reshape层16
        layer_f16 = Dropout(0.2)(layer_f16)
        layer_f17 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f16)    #全连接层11
        layer_f17 = Dropout(0.3)(layer_f17)
        layer_fu18 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f17)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu18)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data
