import numpy as np
main = np.load('main/result.npy')
pr = np.load('pr/result.npy')
alternative = np.load('alternative/result.npy')
def unpack(x):
    return (x.astype(np.uint32) << 16).view(np.float32)
def ordered(x):
    v = x.astype(np.int32)
    return np.where(v & 0x8000, 0xffff-v, v+0x8000)
d = np.abs(ordered(main)-ordered(pr))
x, y = unpack(main).astype(np.float64).ravel(), unpack(pr).astype(np.float64).ravel()
print('lanes', d.size, 'unequal', np.count_nonzero(d), 'alternative_equal', np.array_equal(main, alternative))
print('pcc', np.corrcoef(x, y)[0, 1], 'max_abs', np.max(np.abs(x-y)), 'ulp p50/p95/p99/max', np.percentile(d, [50,95,99,100]))
