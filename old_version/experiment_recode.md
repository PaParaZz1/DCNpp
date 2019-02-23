# experiment record

# runtime between dcn and conv
 - input feature map: 4, 256, 120, 80
 - output feature map: 4, 256, 120, 80
 - conv setting: kernel_size=3, stride=1, padding=1, layers=1
 GPU test: (avg 10 times, GTX 1080TI)
  - avg dcn time: 1.600714
  - avg conv time: 0.000454
  - dcn over conv times: 1678 
 CPU test: (avg 10 times，Inter(R) Xeon(R) Gold 6142 CPU @ 1.60GHz)
  - avg dcn time: 21.743449
  - avg conv time: 0.575270
  - dcn over conv times: 56
