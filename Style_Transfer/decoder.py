from keras.layers import Input, Conv2D, UpSampling2D,Conv2DTranspose

def decoder_layers(inputs, layer):
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(inputs)
    if layer == 1:
        return x
    
    x = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', name='decoder_block4_conv4')(x)
    x = Conv2D(512, (2, 2), activation='relu', padding='same', name='decoder_block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    if layer == 2:
        return x
    
    x = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', name='decoder_block3_conv4')(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    if layer == 3:
        return x
    
    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', name='decoder_block2_conv4')(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    if layer == 4:
        return x
    
    x = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', name='decoder_block1_conv4')(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='decoder_block1_conv1')(x)
    if layer == 5:
        return x

