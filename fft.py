# Copyright 2017 Pontus Rydin, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import time
import requests
import json
import os
import sys
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

granularity = 5 # 5 minute samples

base_url = os.environ['WF_URL']
api_key = os.environ['WF_TOKEN']

metric = sys.argv[1]
plot = len(sys.argv) == 3 and sys.argv[2] == "--plot"

query = 'align({0}m, interpolate(ts("{1}")))'.format(granularity, metric)

start = time.time() - 86400 * 5 # Look at 5 days of data

result = requests.get('{0}/api/v2/chart/api?q={1}&s={2}&g=m'.format(base_url, query, start),
                      headers={"Authorization": "Bearer " + api_key, "Accept": "application/json"})
candidates = []
data = json.loads(result.content)
for object in data["timeseries"]:

    samples = []

    for point in object["data"]:
        samples.append(point[1])

    n = len(samples)
    if n < 2:
        continue

    # Normalized to unity amplitude around mean
    #
    top = np.max(samples)
    bottom = np.min(samples)
    mid = np.average(samples)
    normSamples = (samples - mid)
    normSamples /= top - bottom

    # Calculate power spectrum by squaring the absolute values of the Fourier Transform
    # Interestingly, this happens to be the exact same thing a the absolute value of Fourier
    # transform of the autocorrelation function. One way of interpreting this is that we're
    # obtaining the period of the "ripples" on the autocorrelation function.
    #
    spectrum = np.abs(fft(normSamples))
    spectrum *= spectrum

    # Skip the first 5 samples in the frequency domain to suppress DC interference
    #
    offset = 5
    spectrum = spectrum[offset:]

    # Find the peak
    #
    maxInd = np.argmax(spectrum[:int(n/2)+1])

    # Calculate the total spectral energy. We'll express peaks as a percentage of the
    # total energy.
    #
    energy = np.sum(spectrum)

    # Calculate scaled peak and period
    #
    top = spectrum[maxInd] / energy
    lag = 5 * n / (maxInd + offset)
    if top > 0.1:
        entry = [ top, lag, object["host"] ]
        if plot:
            entry.append(spectrum)
        candidates.append(entry)

best_matches = sorted(candidates, key=lambda match: match[0], reverse=True)
for m in best_matches:
   print("{0},{1},{2}".format(m[2], m[1], m[0]))
   if plot:
       plt.plot(m[3], '-r')
       plt.grid()
       plt.show()


