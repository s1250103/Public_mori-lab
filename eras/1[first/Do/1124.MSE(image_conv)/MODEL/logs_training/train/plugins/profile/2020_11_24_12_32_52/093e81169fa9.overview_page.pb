�	�st�@�st�@!�st�@	�1G��@�1G��@!�1G��@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�st�@���E�?1x` �#@AUܸ��ܰ?IJ�U��?YD�H����?*	������Z@2F
Iterator::Model�t�����?![5��B@)_}<�ݭ�?1}��ʓ4:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]��k�?!�[{<��A@)͐*�WY�?1�\��l�8@:Preprocessing2U
Iterator::Model::ParallelMapV2w�Df.p�?!'A@u>'@)w�Df.p�?1'A@u>'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�7�W���?!g�~T %@)�7�W���?1g�~T %@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�^EF$�?!�A��S/@)�z�΅�?1a3��� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceb��??!9��]�@)b��??19��]�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ꫫ�?!���O@)GW��:{?1M����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapXU/��d�?!��2Tv4@)�ڧ�1u?1333333@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�1G��@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���E�?���E�?!���E�?      ��!       "	x` �#@x` �#@!x` �#@*      ��!       2	Uܸ��ܰ?Uܸ��ܰ?!Uܸ��ܰ?:	J�U��?J�U��?!J�U��?B      ��!       J	D�H����?D�H����?!D�H����?R      ��!       Z	D�H����?D�H����?!D�H����?JGPUY�1G��@b �"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam(H�����?!(H�����?"l
Bgradient_tape/sequential_1/conv_filter/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)Qv��U�?!����7��?"D
(gradient_tape/sequential_1/hidden/MatMulMatMul,%��ğ�?!�(�({�?"6
sequential_1/hidden/MatMulMatMul�G�C��?!�������?"F
*gradient_tape/sequential_1/hidden/MatMul_1MatMul�^�(K}�?!��P�a�?";
sequential_1/conv_filter/Conv2DConv2D̉&��P�?!`OZܓ��?"[
:gradient_tape/sequential_1/max_pooling/MaxPool/MaxPoolGradMaxPoolGrad��R��5�?!�|?{�	�?"M
/gradient_tape/sequential_1/conv_filter/ReluGradReluGrad��ǜ�?!]~a]��?"=
 sequential_1/conv_filter/BiasAddBiasAdd]�>Y�m�?!�QH,ʼ�?"7
sequential_1/conv_filter/ReluRelu��O�l�?!nт+`�?Q      Y@Y�Cc}h4@a9/���S@q���w1@y��G{��?"�
both�Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�28.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�17.468% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 