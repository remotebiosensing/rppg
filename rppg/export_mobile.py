import julius
import torch
import torch.nn as nn

from rppg.nets.CHROM import CHROM
from rppg.nets.GREEN import GREEN
from rppg.nets.ICA import ICA
from rppg.nets.LGI import LGI
from rppg.nets.PBV import PBV
from rppg.nets.PCA import PCA
from rppg.nets.POS import POS
from rppg.nets.SSR import SSR
from scipy import signal
from scipy.sparse import spdiags

'''
    need to verify
    only purpose of test in mobile app with non-DNN methods
    
    test sequence
    1. load non-DNN model from rppg.nets
    2. add preprocess in test class
'''
class clacClass(nn.Module):

    def custom_fft(self,signal):
        N = signal.size(0)
        n = torch.arange(N, dtype=torch.float32).view(N, 1)
        k = torch.arange(N, dtype=torch.float32).view(1, N)

        # 각각의 실수와 허수 부분에 대한 계수를 계산합니다.
        cos_term = torch.cos(-2 * 3.14 * k * n / N)
        sin_term = torch.sin(-2 * 3.14 * k * n / N)

        # 실수 부분과 허수 부분을 계산합니다.
        real_part = torch.matmul(cos_term, signal)
        imag_part = torch.matmul(sin_term, signal)

        return real_part, imag_part

    def custom_rfft(self,signal):
        # FFT를 계산합니다.
        real_part, imag_part = self.custom_fft(signal)

        # 실수 FFT의 대칭성을 이용합니다.
        # N이 짝수인 경우와 홀수인 경우를 다룹니다.
        N = signal.size(0)

        if N % 2 == 0:
            return real_part[:N // 2 + 1], imag_part[:N // 2 + 1]
        else:
            return real_part[:(N + 1) // 2], imag_part[:(N + 1) // 2]
    def calc_hr_torch(self, ppg_signals):

        test_n, sig_length = ppg_signals.shape
        hr_list = torch.empty(test_n)
        ppg_signals = ppg_signals - torch.mean(ppg_signals, dim=-1, keepdim=True)
        N = sig_length

        # Compute frequency and amplitude
        k = torch.arange(N)
        T = N / 30
        freq = k / T

        # Assuming self.custom_rfft returns a tuple (real_part, imag_part)
        real_part, imag_part = self.custom_rfft(ppg_signals)

        amplitude = torch.sqrt(real_part**2 + imag_part**2) / N

        # Find the frequency with the maximum amplitude
        hr_list = freq[torch.argmax(amplitude, dim=-1)] * 60

        return hr_list

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return self.calc_hr_torch(x)
class NormalizeClass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return self.normalize_torch(x)
    def normalize_torch(self,input_val):
        # if type(input_val) != torch.Tensor:
        #     input_val = torch.from_numpy(input_val.copy())
        min = torch.min(input_val, dim=-1, keepdim=True)[0]
        max = torch.max(input_val, dim=-1, keepdim=True)[0]
        return (input_val - min) / (max - min)
class BandPassFilter(torch.nn.Module):
    def __init__(self):
        super(BandPassFilter, self).__init__()
        self.sample_rate = 30
        # 초기 cutoff frequency 값을 설정합니다.
        self.low_cutoff = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.high_cutoff = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def set_cutoff_frequencies(self, low_cutoff, high_cutoff):
        self.low_cutoff.data = torch.tensor(low_cutoff)
        self.high_cutoff.data = torch.tensor(high_cutoff)

    def custom_rfftfreq(self,signals, signal_len, sample_rate):
        # 신호 길이에 따라 주파수 배열 계산
        if signal_len % 2 == 0:
            # 짝수 길이의 신호
            num_freqs = signal_len // 2 + 1
        else:
            # 홀수 길이의 신호
            num_freqs = (signal_len + 1) // 2

        # 주파수 배열 생성
        freqs = torch.arange(0, num_freqs, device=signals.device) * (sample_rate / signal_len)
        return freqs

    def custom_fft(self,signal):
        N = signal.size(0)
        n = torch.arange(N, dtype=torch.float32).view(N, 1)
        k = torch.arange(N, dtype=torch.float32).view(1, N)

        # 각각의 실수와 허수 부분에 대한 계수를 계산합니다.
        cos_term = torch.cos(-2 * 3.14 * k * n / N)
        sin_term = torch.sin(-2 * 3.14 * k * n / N)

        # 실수 부분과 허수 부분을 계산합니다.
        real_part = torch.matmul(cos_term, signal)
        imag_part = torch.matmul(sin_term, signal)

        return real_part, imag_part

    def custom_rfft(self,signal):
        # FFT를 계산합니다.
        real_part, imag_part = self.custom_fft(signal)

        # 실수 FFT의 대칭성을 이용합니다.
        # N이 짝수인 경우와 홀수인 경우를 다룹니다.
        N = signal.size(0)

        if N % 2 == 0:
            return real_part[:N // 2 + 1], imag_part[:N // 2 + 1]
        else:
            return real_part[:(N + 1) // 2], imag_part[:(N + 1) // 2]

    def custom_irfft(self,real_fft, imag_fft, n):
        N = real_fft.size(0)
        n = torch.arange(n, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).view(N, 1)

        # Expand real_fft and imag_fft to match the target size n
        real_fft_expanded = torch.cat((real_fft, real_fft[1:-1].flip(0)), dim=0)
        imag_fft_expanded = torch.cat((imag_fft, -imag_fft[1:-1].flip(0)), dim=0)

        # Calculating the cos and sin terms
        cos_term = torch.cos(2 * 3.14 * k * n / n.size(0))
        sin_term = torch.sin(2 * 3.14 * k * n / n.size(0))

        # Performing the inverse DFT
        signal_real = torch.sum(real_fft_expanded * cos_term - imag_fft_expanded * sin_term, dim=0) / n.size(0)
        return signal_real

    def forward(self, signals):
        self.set_cutoff_frequencies(0.18,0.5)
        b, signal_len = signals.shape
        # signal_list = freq[torch.argmax(amplitude, dim=-1)] * 60
        filtered_signals = torch.empty(signals.shape)
        for i in range(b):
            freq = self.custom_rfftfreq(signals,signal_len, self.sample_rate)
#torch.fft.rfftfreq(signal_len, d=1/self.sample_rate)
            # 복소수 FFT 수행
            real_fft, imag_fft = self.custom_rfft(signals[i])
            band_pass_mask = (freq > self.low_cutoff) & (freq < self.high_cutoff)
            filtered_real_fft = real_fft * band_pass_mask.to(dtype=torch.float32)
            filtered_imag_fft = imag_fft * band_pass_mask.to(dtype=torch.float32)

            # 복소수 텐서 생성
            # 역 FFT 수행을 위한 복소수 텐서 구성
            # 복소수 텐서를 사용하지 않고 실수 부분과 허수 부분을 따로 처리합니다.
            filtered_signal_real = self.custom_irfft(filtered_real_fft, filtered_imag_fft, signal_len)


            # 최종적으로 필터링된 신호는 실수 부분만 사용합니다.
            filtered_signals[i] = filtered_signal_real
        return filtered_signals
class DetrendClass(nn.Module):
    def iterative_inverse(self,matrix, iterations=10):
        # 초기 근사치는 단위 행렬
        inverse_approx = torch.eye(matrix.size(0), device=matrix.device)

        # 반복적인 방법으로 근사치 업데이트
        for _ in range(iterations):
            inverse_approx = 2 * inverse_approx - inverse_approx @ matrix @ inverse_approx

        return inverse_approx
    def detrend_torch(self,signals: torch.Tensor, Lambda: float = 100.0) -> torch.Tensor:
        """
        Detrend 1D signals with diagonal matrix D, using torch batch matrix multiplication.

        :param signals: Signals with linear trend, expected to be a PyTorch tensor.
        :param Lambda: A scalar value used in detrending calculation, expected to be a float.
        :return: Detrended signals as a PyTorch tensor.
        """
        test_n, length = signals.shape

        # Constructing the D matrix for detrending
        H = torch.eye(length).to(signals.device)
        ones = torch.ones(length - 2).to(signals.device)
        diag1 = torch.zeros((length - 2, length), device=signals.device)
        diag2 = torch.zeros((length - 2, length), device=signals.device)
        diag3 = torch.zeros((length - 2, length), device=signals.device)

        # diag1 채우기
        for i in range(length - 2):
            diag1[i, i] = 1
            if i + 1 < length - 2:
                diag1[i, i + 1] = 1

        # diag2 채우기
        for i in range(length - 2):
            diag2[i, i + 1] = -2

        # diag3 채우기
        for i in range(length - 2):
            if i + 2 < length:
                diag3[i, i + 2] = 1

        D = diag1 + diag2 + diag3

        # Convert Lambda to a tensor
        #Lambda_tensor = torch.tensor(Lambda).to(signals.device)

        # Detrending calculation
        I = torch.eye(H.size(0), device=H.device)

        # 선형 시스템 해결을 사용하여 역행렬 계산
        inv_matrix = self.iterative_inverse(H + (Lambda ** 2) * torch.transpose(D, 0, 1) @ D)

        # 원래 연산 수행
        detrended_signal = torch.bmm(signals.unsqueeze(1), (H - inv_matrix).expand(test_n, -1, -1)).squeeze()
        return detrended_signal

    def __init__(self):
        super().__init__()

    def forward(self,input_val):
        return self.detrend_torch(input_val)
class MobileClass(nn.Module):

    def detrend_torch(self,signals: torch.Tensor, Lambda: float = 100.0) -> torch.Tensor:
        """
        Detrend 1D signals with diagonal matrix D, using torch batch matrix multiplication.

        :param signals: Signals with linear trend, expected to be a PyTorch tensor.
        :param Lambda: A scalar value used in detrending calculation, expected to be a float.
        :return: Detrended signals as a PyTorch tensor.
        """
        test_n, length = signals.shape

        # Constructing the D matrix for detrending
        H = torch.eye(length).to(signals.device)
        ones = torch.ones(length - 2).to(signals.device)
        #diag1 = torch.cat((torch.diag(ones), torch.zeros((length - 2, 2), device=signals.device)), dim=-1)
        #diag2 = torch.cat((torch.zeros((length - 2, 1), device=signals.device), torch.diag(-2 * ones),
        #                   torch.zeros((length - 2, 1), device=signals.device)), dim=-1)
        #diag3 = torch.cat((torch.zeros((length - 2, 2), device=signals.device), torch.diag(ones)), dim=-1)

        # diag1, diag2, diag3에 대한 영행렬 생성
        diag1 = torch.zeros((length - 2, length), device=signals.device)
        diag2 = torch.zeros((length - 2, length), device=signals.device)
        diag3 = torch.zeros((length - 2, length), device=signals.device)

        # diag1 채우기
        for i in range(length - 2):
            diag1[i, i] = 1
            if i + 1 < length - 2:
                diag1[i, i + 1] = 1

        # diag2 채우기
        for i in range(length - 2):
            diag2[i, i + 1] = -2

        # diag3 채우기
        for i in range(length - 2):
            if i + 2 < length:
                diag3[i, i + 2] = 1
        D = diag1 + diag2 + diag3

        # Convert Lambda to a tensor
        #Lambda_tensor = torch.tensor(Lambda).to(signals.device)

        # Detrending calculation
        detrended_signal = torch.bmm(signals.unsqueeze(1),
                                     (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).expand(test_n,
                                                                                                              -1,
                                                                                                              -1)).squeeze()
        return detrended_signal

    def normalize_torch(self,input_val):
        # if type(input_val) != torch.Tensor:
        #     input_val = torch.from_numpy(input_val.copy())
        min = torch.min(input_val, dim=-1, keepdim=True)[0]
        max = torch.max(input_val, dim=-1, keepdim=True)[0]
        return (input_val - min) / (max - min)

    def calc_hr_torch(self, ppg_signals):

        test_n, sig_length = ppg_signals.shape
        hr_list = torch.empty(test_n)
        ppg_signals = ppg_signals - torch.mean(ppg_signals, dim=-1, keepdim=True)
        N = sig_length

        # Compute frequency and amplitude
        k = torch.arange(N)
        T = N / 30
        freq = k / T
        amplitude = torch.abs(torch.fft.rfft(ppg_signals, n=N, dim=-1)) / N

        # Find the frequency with the maximum amplitude
        hr_list = freq[torch.argmax(amplitude, dim=-1)] * 60

        return hr_list
    def __init__(self, model_name = "POS"):
        super().__init__()
        if model_name == "GREEN":
            self.model = GREEN()
        elif model_name == "POS":
            self.model = POS()
        elif model_name == "CHROM":
            self.model = CHROM()
        elif model_name == "LGI":
            self.model = LGI()
        elif model_name == "PBV":
            self.model = PBV()
        elif model_name == "SSR":
            self.model = SSR()
        elif model_name == "PCA":
            self.model = PCA()
        elif model_name == "ICA":
            self.model = ICA()

        #self.bpf_hr = BandPassFilter(30)
        #self.bpf_hr.set_cutoff_frequencies(0.75, 3)

        #self.bpf_rr = BandPassFilter(30)
        #self.bpf_rr.set_cutoff_frequencies(0.18,0.5)

    def forward(self,x):
        rppg = self.model(x)
        #rppg = self.detrend_torch(rppg)

        #rppg_hr = self.normalize_torch(self.bpf_hr(rppg))
        #rppg_rr = self.normalize_torch(self.bpf_rr(rppg))

        #hr = self.calc_hr_torch( rppg_hr)
        #rr = self.calc_hr_torch(rppg_rr)

        return torch.tensor([70, 15, 2.0, 98, 30, 117, 68])
from torch.utils.mobile_optimizer import optimize_for_mobile
if __name__ == "__main__":
    print(torch.has_lapack)
    # x = torch.randn(2, 3, 300, 50, 50)
    rppg = torch.randn(2,300)
    # mobileclass = MobileClass(model_name="POS")
    # torch.save(mobileclass, './pos.pt')
    # traced_script_module = torch.jit.trace(mobileclass, x)
    #print(traced_script_module)
    #detrendclass = DetrendClass()
    # hrbpfclass = BandPassFilter()
    # normalizeclass = NormalizeClass()
    calcclass = clacClass()

    torch.onnx.export(calcclass,  # 실행될 모델
                      rppg,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "calc.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output']
                      )


    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("model.ptl")
    # loaded = torch.jit.load('model.ptl')
    # print(loaded(x))