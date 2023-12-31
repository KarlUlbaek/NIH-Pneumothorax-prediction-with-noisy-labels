# this is just some stupid copy past from the printed output list found in one of the cells in find_outliers notebook

# from kfold_mix_exp_01
# given scaling of 0.20000: amount of outliers is 1.99%
# combined
#  [[8046  168]
#  [   0  244]]
# 168 found (all positive samples/labels), 60male, 108female
OUTLIER_IDS_MIXUP_2PERCENT = [3,164,4315,80,4253,16,4366,71,223,4377,4278,171,36,91,4290,4230,167,55,4390,204,4284,4274,
                              176,211,67,4273,41,4240,157,95,98,175,4245,4401,4330,114,163,199,4405,173,181,135,136,182,
                              4376,88,4393,4238,24,4317,162,200,105,4370,4355,168,63,4280,192,187,212,2,126,99,4334,4367,
                              35,215,4322,4395,62,148,147,21,58,156,22,230,128,4316,4259,29,207,37,170,46,166,4241,4251,
                              100,195,56,33,185,4321,228,178,214,4406,227,4359,34,107,4308,4270,201,133,4297,5,4347,4309,
                              84,43,26,4398,68,12,4256,4279,4323,4380,4319,4264,15,79,4277,51,81,28,4295,4266,102,4328,
                              27,154,4231,4294,87,4336,61,125,90,103,13,4344,142,4282,174,131,117,198,4304,110,4285,172,
                              39,4354,129,108,121,4384,124,60,123,53,179,4265,218]

# given scaling of 0.08500: amount of outliers is 1.00%
# combined
#  [[8046   85]
#  [   0  327]]
# 85 found (all positive samples/labels), 28male, 57female
OUTLIER_IDS_MIXUP_1PERCENT = [164,16,71,4377,4278,171,36,4230,4274,176,211,41,4240,157,98,175,4245,4401,4330,114,4405,
                              173,136,182,88,4393,4317,162,105,63,187,212,2,99,4367,4322,62,148,147,22,128,4316,207,
                              37,4251,195,56,33,185,4321,214,227,4359,34,107,133,5,4309,84,43,68,4256,4323,4319,4264,
                              15,51,81,28,4266,4328,154,4231,87,103,4344,142,174,198,4304,121,4384,124,123,53]

# s = 0.0006, 1%
# 89 found (all positive samples/labels), 33male, 56female
# OUTLIER_IDS_PLAIN_1PERCENT = [215,125,208,4278,4238,67,53,62,166,21,32,4241,128,88,4230,182,185,4244,4392,4298,4265,4387,
#                               97,55,4261,173,113,107,28,229,4260,4406,63,87,227,121,43,90,214,4275,4317,199,20,136,207,
#                               194,81,4297,141,142,4279,4377,4253,4360,6,140,4309,12,103,4266,4270,4251,195,4381,4390,46,
#                               84,4304,68,4321,228,4280,179,3,37,71,16,5,34,56,4330,4384,4315,120,4264,154,175,4367,15]
#

# given scaling of 0.16860: amount of outliers is 1.99%
# combined:
#  [[1341   28]
#  [   0   40]]
VAL_OUTLIER_IDS_MIX_2PERCENT = [0,3,4,7,8,11,12,14,15,18,22,23,24,27,30,33,708,710,711,712,715,716,717,719,721,723,726,729]
#

# given scaling of 0.06100: amount of outliers is 0.99%
# combined:
#  [[1341   14]
#  [   0   54]]
# 14 found (all positive samples/labels), 8male, 6female
VAL_OUTLIER_IDS_MIX_1PERCENT = [0, 3, 7, 22, 24, 33, 710, 715, 716, 717, 721, 723, 726, 729]


#s = 0.001686
# combined:
#  [[1341   30]
#  [   0   38]]
#VAL_OUTLIER_IDS_PLAIN_2PERCENT = [0,3,7,8,9,11,14,16,18,22,23,24,27,30,33,38,705,708,710,711,712,713,714,715,716,717,719,721,723,726]



# GMM MIXUP:
GMM_MIXUP = [4309,  195, 4251 ,4328 ,4253  ,179  ,151  , 15 ,181  ,105]
#num out of [35. 35. 35. 35. 35. 35. 35. 35. 35. 35.]


if __name__ == "__main__":
   import numpy as np
   # print("overlap between 2 sets:",
   #       np.isin(np.asarray(OUTLIER_IDS_MIXUP_2PERCENT), np.asarray(OUTLIER_IDS_PLAIN_1PERCENT)).mean())
   #
   # print("VAL overlap between 2 sets:",
   #    np.isin(np.asarray(VAL_OUTLIER_IDS_MIX_2PERCENT), np.asarray(VAL_OUTLIER_IDS_PLAIN_2PERCENT)).mean())

   print("overlap GMM_mix_up and CL_mixup:",
      np.isin(np.asarray(GMM_MIXUP), np.asarray(OUTLIER_IDS_MIXUP_2PERCENT)).mean())