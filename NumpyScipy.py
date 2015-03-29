import math
import cmath

def hanning(m):
    return [0.5-0.5*math.cos(2*math.pi*n/(m-1)) for n in range(m)] if m!=1 else [1.0]

def convolve(arr1, arr2, mode = 'full'):
    length = len(arr1)+len(arr2)-1
    res = [0.0]*length
    for n in range(length):
        temp = []
        for i in range(len(arr1)):
            if (n-i) < len(arr2) and (n-i >= 0):
                temp.append(arr1[i]*arr2[n-i])
        res[n] = sum(temp)
    if mode == 'full':
        return res
    elif mode == 'same':
        while len(res) > max(len(arr1), len(arr2)):
            if len(res) > max(len(arr1), len(arr2)):
                res.pop(-1)
            if len(res) > max(len(arr1), len(arr2)):
                res.pop(0) 
        return res
    elif mode == 'valid':
        return res[min(len(arr1),len(arr2))-1:max(len(arr1),len(arr2))]    
    else :
        return None

def arange(start, end=None,step = 1):
    if end == None:
        end = start
        start = 0
    res = []
    while start < end:
        res.append(float(start))
        start += step    
    return res

def fft(arr, n=None, axis = -1, overwrite_x = 0):
    res = [0.0]* len(arr)
    for j in range(len(arr)):
        temp = []
        for k in arange(len(arr)):
            temp.append(arr[int(k)]*cmath.exp(-cmath.sqrt(-1)*j*k*2*cmath.pi/len(arr)))
        res[j] = sum(temp)
    return res

def ifft(arr, n=None, axis = -1, overwrite_x = 0):
    res = [0.0]* len(arr)
    for j in range(len(arr)):
        temp = []
        for k in arange(len(arr)):
            temp.append(arr[int(k)]*cmath.exp(cmath.sqrt(-1)*j*k*2*cmath.pi/len(arr)))
        res[j] = sum(temp)/len(temp)
    return res

def fftfreq(n, d=1.0):
    return div(arange(n/2)+arange(-n/2, 0), (d*n)) if n%2==0 else div(arange(n/2+1)+arange(-n/2+1, 0), (d*n))

def abs_iter(iterable):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(abs_iter(item))
        else:
            result.append(abs(item))
    return result

def abs_complex(arr):
    return [abs(elem) for elem in arr]

def mult(arr, num):
    return [num*elem for elem in arr]

def minus(arr, num):
    result = []
    for item in arr:
        if isinstance(item, list):
            result.append(minus(item, num))
        else:
            result.append(item-num)
    return result

def plus(arr, num):
    return [num+elem for elem in arr]
    
def plus_arr(arr1, arr2):
    return [arr1[i]+arr2[i] for i in range(len(arr1))] if len(arr1) == len(arr2) else None

def plus_iter(iterable, num):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(plus_iter(item, num))
        else:
            result.append(item+num)
    return result

def minus_arr(arr1, arr2):
    return [arr1[i]-arr2[i] for i in range(len(arr1))] if len(arr1) == len(arr2) else None

def mult_arr(arr1, arr2):
    return [arr1[i]*arr2[i] for i in range(len(arr1))] if len(arr1) == len(arr2) else None

def mult_iter(iterable, num):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(mult_iter(item, num))
        else:
            result.append(item*num)
    return result

def div_arr(arr1, arr2):
    return [arr1[i]/arr2[i] for i in range(len(arr1))] if len(arr1) == len(arr2) else None

def real_arr(arr):
    return [elem.real for elem in arr]

def div(arr, num):
    return [elem/num for elem in arr]

def div_iter(iterable, num):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(div_iter(item, num))
        else:
            result.append(item/num)
    return result

def power(arr, num):
    return [pow(elem, num) for elem in arr]

def power_iter(iterable, num):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(power_iter(item, num))
        else:
            result.append(math.pow(item, num))
    return result

def exp(arr):
    return [cmath.exp(elem) for elem in arr]

def log10(arr):
    return [cmath.log10(elem) for elem in arr]
    
def sqrt(arr):
    return [cmath.sqrt(elem) for elem in arr]

def around(a, decimals = 0):
    return [round(elem, decimals) for elem in a]

def argmax(arr):
    return arr.index(max(arr))

def getMin(iterable):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(getMin(item))
        else:
            result.append(item)
    return min(result)

def getMax(iterable):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(getMax(item))
        else:
            result.append(item)
    return max(result)

def getSum(iterable):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(getSum(item))
        else:
            result.append(item) 
    return sum(result)

def getNum(iterable):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(getNum(item))
        else:
            result.append(1) 
    return sum(result)

#print getSum([1,2,[3,4], [5,[6,7]]])
#print getNum([1,2,[3,4], [5,[6,7]]])
#print getSum([1,2, [5,[6,7]]])
#print getSum([1,2, [5,[6,7]],[[[1]]]])
def getMean(iterable):
    return (getSum(iterable)+0.0) / getNum(iterable)

def getMedian(iterable):
    sort_iterable = sorted(iterable)
    length = len(sort_iterable)
    if length%2 == 1:
        return sort_iterable[length/2]
    else:
        return (sort_iterable[length/2-1]+sort_iterable[length/2])/2.0
 
def std(arr):
    return math.sqrt(getMean(power_iter((abs_iter(minus(arr, getMean(arr)))),2)))

def hstack((a,b)):
    return [a[i]+b[i] for i in range(len(a))]

def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def scoreatpercentile(arr, num):
    return min(arr)+(max(arr)-min(arr))*num/100.0

def skew(arr):
    avg = getMean(arr)
    resCB = []
    resSQ = []
    for i in arr:
        resCB.append(math.pow(i-avg, 3))
        resSQ.append(math.pow(i-avg, 2))
    up = sum(resCB)/len(arr)    
    down = math.pow(sum(resSQ)/len(arr), 1.5)
    return up/down if down!=0 else 0.0

def kurtosis(arr):
    avg = getMean(arr)
    resFO = []
    resSQ = []
    for i in arr:
        resFO.append(math.pow(i-avg, 4))
        resSQ.append(math.pow(i-avg, 2))
    up = sum(resFO)/len(arr)    
    down = math.pow(sum(resSQ)/len(arr), 2)
    return up/down-3 if down!=0 else -3.0

def hilbert(x, N=None, axis=-1):
    if N is None:
        N = len(x)
    Xf = fft(x, N, axis=axis)
    h = [0.0]*N
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = [2]*((N / 2)-1)
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = [2]* ((N+1)/2-1)

    x = ifft(mult_arr(Xf , h), axis=axis)    
    return x

def clip(a, a_min, a_max, out=None):
    res = []
    for elem in a:
        if a < a_min:
            res.append(a_min)
        elif a > a_max:
            res.append(a_max)
        else:
            res.append(a)
    return res

def clip_iter(iterable, a_min, a_max):
    result = []
    for item in iterable:
        if isinstance(item, list):
            result.append(clip_iter(item, a_min, a_max))
        else:
            if item < a_min:
                result.append(a_min)
            elif item > a_max:
                result.append(a_max)
            else:
                result.append(item)
    return result

def transpose(arr):
    return [list(i) for i in zip(*arr)]

def zeros(num):
    return [0.0 for n in range(num)]

def zeros2D((a,b)):
    res = []
    for i in range(a):
        res.append([])
        for j in range(b):
            res[i].append(0.0)          
    return res

def zeros3D((a,b,c)):
    res = []
    for i in range(a):
        res.append([])
        for j in range(b):
            res[i].append([])
            for k in range(c):
                res[i][j].append(0.0)
    return res

def appear(arr, a_min, a_max):
    count = 0
    for a in arr:
        if a >=a_min and a<a_max:
            count += 1
    return count

def histogram(arr, bins=256, normed = True):
    li = flatten(arr)
    minval = min(li)
    maxval = max(li)
    interval = (maxval-minval)/(bins+0.0)
    tempval = minval
    hist = []
    bin_edges = []
    count = 0
    while tempval < maxval+ interval and count<=bins:
        bin_edges.append(tempval)
        tempval += interval
        count += 1

    for r in range(bins):
        count = appear(li, bin_edges[r], bin_edges[r+1])  
        hist.append((count+0.0)/len(li)/interval)
    return hist, bin_edges

def interp(li, xp, fp):
    res = []
    for elem in li:
        if elem <= xp[0]:
            res.append(fp[0])
        elif elem >= xp[-1]:
            res.append(fp[-1])
        else:
            for xr in range(len(xp)-1):
                if elem == xp[xr]:
                    res.append(fp[xr])
                    break
                elif elem <xp[xr+1] and elem >xp[xr]:
                    res.append( (fp[xr+1]-fp[xr])*(elem-xp[xr])/(xp[xr+1]-xp[xr]) + fp[xr] )
                    break
                
    return res


def test_numpyFunc():
    print hanning(1)
   # print np.hanning(1)
    
    print hanning(12)
    #print np.hanning(12)
    
    print convolve([1,2,3,4,5], [0,1,0.5])
   # print np.convolve([1,2,3,4,5], [0,1,0.5])
    
    print convolve([0,1,0.5], [1,2,3,4])
    #print np.convolve([0,1,0.5], [1,2,3,4])
    
    print convolve([1,2,3,100], [0,1,0.5])
   # print np.convolve([1,2,3,100], [0,1,0.5])
    
    print convolve([1,2,3,4], [1,2,3,4])
    #print np.convolve([1,2,3,4], [1,2,3,4])
    
    print convolve([1,2,3,4,5], [0,1,0.5], 'same')
    #print np.convolve([1,2,3,4,5], [0,1,0.5],'same')
    
    print convolve([1,2,3,4,5], [0,1], 'same')
    #print np.convolve([1,2,3,4,5], [0,1], 'same')
    
    print convolve([1,2,3,4,5], [0], 'same')
    #print np.convolve([1,2,3,4,5], [0], 'same')   
    
    print convolve([1,2,3,4], [1,2,3,4], 'same')
    #print np.convolve([1,2,3,4], [1,2,3,4], 'same') 
    
    #print np.convolve([0,1,0.5], [1,2,3,4], 'valid')
    print convolve([0,1,0.5], [1,2,3,4], 'valid')
    
    #print np.convolve([0,1,0.5], [1,2,3], 'valid')
    print convolve([0,1,0.5], [1,2,3], 'valid')
    
    #print np.convolve([1,2,3], [0,1,0.5], 'valid')
    print convolve([1,2,3], [0,1,0.5], 'valid')
    
    #print np.convolve([1,2,3,4], [0,1,0.5], 'valid')
    print convolve([1,2,3,4], [0,1,0.5], 'valid')
    
    print convolve([1,2,3,4], [1,2,3,4], 'valid')
    #print np.convolve([1,2,3,4], [1,2,3,4], 'valid')
    
    print fft([1,2,3])
    print abs_complex(fft([1,2,3]))
    
    print fft([1,2,3,4,5,6,7,8,9,10])
    print abs_complex(fft([1,2,3,4,5,6,7,8,9,10]))
    
    print arange(3)
    print arange(3.0)
    print arange(3, 7)
    print arange(3,7,2)

＃test_numpyFunc()