import tensorflow as tf

class GridSample(tf.keras.layers.Layer):
    def __init__(self,mode = "bilinear", padding_mode = "zeros", align_corners=False):
        super(GridSample, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.height = None
        self.width = None
        self.datatype = None
        self.batch_size = None
        self.channel = None

    def compute_output_shape(self, input_shape):

        return [None, self.height, self.width, self.channels]

    def build(self, input_shape):
        
        coord = tf.zeros(shape=input_shape[1])
        x , y = tf.split(coord, 2, 3)
        self.coord_x = tf.Variable(initial_value=x, trainable=False)
        self.coord_y = tf.Variable(initial_value=y, trainable=False)
        self.outputs = tf.Variable(initial_value=tf.zeros(shape=(self.batch_size, self.height,self.width, self.channel)),trainable=False)
        
        print(input_shape[0])


    def set_attributes(self, inputs, grid):

        self.batch_size= grid.shape[0]
        self.height = grid.shape[1]
        self.width = grid.shape[2]
        self.datatype = inputs.dtype
        self.channel = inputs.shape[3]



    def call(self, inputs):

        inputs, grid = inputs[0], inputs[1]
        self.set_attributes(inputs, grid)

        if (self.height, self.width) != (inputs.shape[1], inputs.shape[2]):
            raise Exception(" input and grid dimensions are not matching.")
        
        if self.mode == "bilinear":
            return self.bilinear_interpolate(inputs, grid)

    def get_config(self):
        config = super(GridSample, self).get_config()
        config.update({"height": self.height, "width":self.width})
        return config

    
    def unnormalize(self,x,y):
        if self.align_corners:
            x = ((x + 1) / 2.0) * (tf.cast(self.width, dtype=self.datatype) - 1)
            y = ((y + 1) / 2.0) * (tf.cast(self.height, dtype=self.datatype)- 1)

        else:
            x = (((x + 1) * tf.cast(self.width, dtype=self.datatype)) - 1) / 2.0
            y = (((y + 1) * tf.cast(self.height, dtype=self.datatype)) - 1) / 2.0
 

        return self.round_tensor(x,y)



    
    def round_tensor(self, x, y):

        #try to inialise it elsewhere
        # assign_add over here.
        self.coord_x.assign_add(x)
        self.coord_y.assign_add(y) 

        for n in tf.range(self.batch_size):
            for h in tf.range(self.height):
                for w in tf.range(self.width):

                    if tf.math.ceil(self.coord_x[n,h,w,0]) - self.coord_x[n,h,w,0] < 0.00001:
                        update = tf.expand_dims(tf.math.ceil(self.coord_x[n,h,w,0]),axis=0)
                        self.coord_x.scatter_nd_update(indices = [[n,h,w,0]],updates=update)
        
                    if tf.math.ceil(self.coord_y[n,h,w,0]) - self.coord_y[n,h,w,0] < 0.00001:
                        update = tf.expand_dims(tf.math.ceil(self.coord_y[n,h,w,0]),axis=0)
                        self.coord_y.scatter_nd_update(indices = [[n,h,w,0]],updates=update)
        return self.coord_x, self.coord_y

        


    def bilinear_interpolate(self, inputs, grid):
        
        with tf.name_scope("Unnormalization"):

            xs, ys = tf.split(grid , num_or_size_splits=2, axis=3)

            # self.coord_x.assign_add(xs)
            # self.coord_y.assign_add(ys) 
            x, y = self.unnormalize(xs, ys)
            
        
        with tf.name_scope("Weights"):

            x1 = tf.math.floor(self.coord_x)
            x2 = x1 + 1
            y1 = tf.math.floor(self.coord_y)
            y2 = y1 + 1

    
            WA = (x2 - x) * (y2 - y)
            WB = (x2 - x) * (y - y1)
            WC = (x - x1) * (y2 - y)
            WD = (x - x1) * (y - y1)     
        
        with tf.name_scope("PixelIntensites"):

            A = self.get_pixel(inputs, tf.cast(x1,dtype=tf.int32), tf.cast(y1, dtype=tf.int32))
            A = tf.convert_to_tensor(A)
            B = self.get_pixel(inputs, tf.cast(x1,dtype=tf.int32), tf.cast(y2,dtype=tf.int32))
            B = tf.convert_to_tensor(B)
            C = self.get_pixel(inputs, tf.cast(x2,dtype=tf.int32), tf.cast(y1,dtype=tf.int32))
            C = tf.convert_to_tensor(C)
            D = self.get_pixel(inputs, tf.cast(x2,dtype=tf.int32), tf.cast(y2,dtype=tf.int32))
            D = tf.convert_to_tensor(D)

        return tf.math.add_n([WA * A + WB * B + WC * C + WD * D])        
        


    def get_pixel(self, inputs, x1, y1):
        # outputs = tf.zeros_like(inputs)
        # outputs = tf.Variable(outputs)
        
        for n in tf.range(self.batch_size):
            for h in tf.range(self.height):
                for w in tf.range(self.width):
                    for c in tf.range(self.channel):
                        x = x1[n,h,w,0]
                        y = y1[n,h,w,0]
                        if x >=0 and x < self.width and y >=0 and y < self.height:

                            update = tf.expand_dims(inputs[n,y,x,c],axis=0)
                            self.outputs.scatter_nd_update(indices=[[n,h,w,c]],updates=update)

                        else:
                            self.outputs.scatter_nd_update(indices=[[n,h,w,c]],updates= tf.constant([0.0],dtype=self.datatype))

        return self.outputs











