import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from IPython.display import Audio

# Ở HW1, ta dùng chuỗi bit (ví dụ: '101') 
# Ở HW này, ta sẽ dùng list các bit (ví dụ: [1, 0, 1]), vì như này sẽ dễ dàng hơn
# cho việc nhúng và rút trích trong HW này
def convert_s2bits(s):
    '''
    Chuyển chuỗi ký tự (ascii) thành list các bit.
    '''
    bits = []
    for c in s:
        c_bits = []
        for bit in list(bin(ord(c))[2:].zfill(8)):
            c_bits.append(int(bit))
        bits.extend(c_bits)
    return bits

def convert_bits2s(bits):
    '''
    Chuyển list các bit thành chuỗi ký tự (ascii).
    '''
    s = ''
    for i in range(0, len(bits), 8):
        c_bits = ''
        for bit in bits[i:i+8]:
            c_bits = c_bits + str(bit)
        c = chr(int(c_bits, 2))
        s = s + c
    return s

def embed(
    msg_file, cover_aud_file, num_segments, time_delta0, time_delta1, 
    stego_aud_file):
    '''
    Nhúng tin mật trên âm thanh bằng phương pháp echo (bạn xem file slide lý 
    thuyết "07..."). Khi nhúng thì ta vẫn dùng chiêu thêm đuôi "100..." vào 
    message bits như ở các HW trước. Khi tạo sóng âm echo thì ta sẽ không dùng
    tham số "decay rate" như ở trang 17 của slide lý thuyết (nói một cách khác: 
    decay rate = 1).
    
    Các tham số:
        msg_file (str): Tên file chứa tin mật.
        cover_aud_file (str): Tên file chứa cover audio.
        num_segments (int): Số lượng đoạn sóng dùng để nhúng (mỗi đoạn sẽ nhúng một bit).
        time_delta0 (float): Độ trễ (giây) của echo ứng với bit 0.
        time_delta1 (float): Độ trễ (giây) của echo ứng với bit 1.
        stego_aud_file (str): Tên file chứa stego audio (kết quả sau khi nhúng).       
    Giá trị trả về:
        bool: True nếu nhúng thành không, False nếu không đủ chỗ để nhúng. 
    '''
    # YOUR CODE HERE
    with open(msg_file, 'r') as f:
        msg = f.read()
    
    msg_bits = convert_s2bits(msg)

    msg_size = len(msg_bits)
    if(msg_size + 1 > num_segments):
        return False
    
    msg_bits.extend('1' + '0' * (num_segments - msg_size - 1))
    print(msg_bits)
    rate, cover_samples = wavfile.read(cover_aud_file)
    cover_samples = np.floor_divide(cover_samples, 2)

    delay0 = int(np.floor(time_delta0 * rate))
    delay1 = int(np.floor(time_delta1 * rate))

    echo0 = np.zeros_like(cover_samples)
    echo1 = np.zeros_like(cover_samples)
    echo0[delay0:] = cover_samples[:-delay0]
    echo1[delay1:] = cover_samples[:-delay1]

    sample_size = int(np.floor(cover_samples.size / num_segments))

    samples_echo0 = cover_samples + echo0
    samples_echo1 = cover_samples + echo1
    mixer0 = [1 - int(bit) for bit in msg_bits]  
    mixer1 = [int(bit) for bit in msg_bits]   
    stego_samples = np.empty_like(cover_samples)

    stego_samples[sample_size*num_segments:] = cover_samples[sample_size * num_segments:]
    for i in range(num_segments):
        idx = sample_size * i
        stego_mini = samples_echo0[idx:idx + sample_size] * mixer0[i] + samples_echo1[idx:idx + sample_size] * mixer1[i]
        stego_samples[idx:idx + sample_size] = stego_mini
    wavfile.write(stego_aud_file, rate, stego_samples)
    return True

def extract(
    stego_aud_file, num_segments, time_delta0, time_delta1, 
    extr_msg_file):
    '''
    Rút trích tin mật đã được nhúng trên âm thanh bằng phương pháp echo.
    
    Các tham số:
        stego_aud_file (str): Tên file chứa stego audio.
        num_segments (int): Số lượng đoạn sóng dùng để nhúng (mỗi đoạn sẽ nhúng một bit).
        time_delta0 (float): Độ trễ (giây) của echo ứng với bit 0.
        time_delta1 (float): Độ trễ (giây) của echo ứng với bit 1.
        extr_msg_file (str): Tên file chứa tin mật được rút trích.
    '''
    # YOUR CODE HERE
    rate, samples = wavfile.read(stego_aud_file)
    n = samples.shape[0]
    n_samples_per_seg = int(n / num_segments)
    extr_msg_bits = []
    # chia 64 đoạn
    samples_64 = samples[:-(n - n_samples_per_seg * num_segments)].reshape((num_segments, n_samples_per_seg))
    for samples_seg in samples_64:
        temp = (samples_seg - samples_seg.mean()) / (samples_seg.var() ** 0.5) 
        k0 = int(time_delta0*rate) - 1
        k1 = int(time_delta1*rate) - 1
        
        ac0 = np.mean(temp[:-k0] * temp[k0:])
        ac1 = np.mean(temp[:-k1] * temp[k1:])
        if(ac0 > ac1):
            extr_msg_bits.append(0)
        else:
            extr_msg_bits.append(1)
    res = len(extr_msg_bits) - 1 - extr_msg_bits[::-1].index(1)
    extr_msg_bits = extr_msg_bits[:res]
    print(extr_msg_bits)
    extracted_message = convert_bits2s(extr_msg_bits)
    with open(extr_msg_file, 'w', encoding='utf-8') as f:
        f.write(extracted_message)


# TEST
extract(
    'correct_stego.wav', num_segments=64, time_delta0=0.03, time_delta1=0.031,
    extr_msg_file='extr_msg.txt')
with open('extr_msg.txt', 'r') as f:
    extr_msg = f.read()
with open('correct_extr_msg.txt', 'r') as f:
    correct_extr_msg = f.read()
assert extr_msg == correct_extr_msg
# Để ý là có một ký tự rút trích bị sai