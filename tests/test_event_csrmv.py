# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import brainstate as bst
import braintaichi as bti


def test_example1():
  events = bst.random.random((1000,)) < 0.1

  # Create a sparse matrix
  r = bti.jitc_event_mv_prob_homo(events, 1., conn_prob=0.1, shape=(1000, 1000), seed=123)
  print(r.shape)
  print(r)

