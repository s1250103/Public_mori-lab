�	���p%@���p%@!���p%@	mUd�d@mUd�d@!mUd�d@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���p%@�G�z�?1�1˞$@A$_	�Į�?Ic���@Y�y�3MX�?*	�$���a@2F
Iterator::Model2���J�?!m1�I!�B@)�Y�$��?1�hwe8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�n�EE��?!aWmh��>@)��kCŠ?1_���6@:Preprocessing2U
Iterator::Model::ParallelMapV2ADj��4�?!G��\z**@)ADj��4�?1G��\z**@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate� %̴�?!7.t�H<4@)�%�`6�?1�!�j!s'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice������?!w:pp!@)������?1w:pp!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�M�g\�?!Ēy�@)�M�g\�?1Ēy�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��K�?!����hO@)�a0�̅?1|��F��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�:���?!d
�D�8@)aU��N�y?1��W�k@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�21.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t13.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9mUd�d@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�G�z�?�G�z�?!�G�z�?      ��!       "	�1˞$@�1˞$@!�1˞$@*      ��!       2	$_	�Į�?$_	�Į�?!$_	�Į�?:	c���@c���@!c���@B      ��!       J	�y�3MX�?�y�3MX�?!�y�3MX�?R      ��!       Z	�y�3MX�?�y�3MX�?!�y�3MX�?JGPUYmUd�d@b �"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamM'<���?!M'<���?"B
&gradient_tape/sequential/hidden/MatMulMatMull�pj7i�?!�i��"��?"4
sequential/hidden/MatMulMatMul���q3��?!W6Ȱ�n�?"D
(gradient_tape/sequential/hidden/MatMul_1MatMul��폥]�?!��a
��?"j
@gradient_tape/sequential/conv_filter/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterp��[�?!�&|'���?"9
sequential/conv_filter/Conv2DConv2Dl��o�L�?!�Ew�?"Y
8gradient_tape/sequential/max_pooling/MaxPool/MaxPoolGradMaxPoolGradeS���?!I��d �?";
sequential/conv_filter/BiasAddBiasAdd�E���9�?!y�
�	�?"K
-gradient_tape/sequential/conv_filter/ReluGradReluGrad)��؈t�?!�&rQt��?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits-���?!b���B��?Q      Y@Ydp>�c8@a�cp>�R@q��"��%@yt�b�Ho�?"�
both�Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�21.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t13.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�10.9258% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 