#include "backprop.h"
#include <math.h>
#include <stdio.h>

float bpnn_train_kernel(
    float input_units[LAYERSIZE + 1],
    float hidden_units[HIDDEN_SIZE + 1],
    float output_units[OUTPUT_SIZE + 1],
    float hidden_delta[HIDDEN_SIZE + 1],
    float output_delta[OUTPUT_SIZE + 1],
    float target[OUTPUT_SIZE + 1],
    float input_weights[LAYERSIZE + 1][1 + HIDDEN_SIZE],
    float hidden_weights[HIDDEN_SIZE + 1][OUTPUT_SIZE + 1],
    float input_prev_weights[LAYERSIZE + 1][HIDDEN_SIZE + 1],
    float hidden_prev_weights[HIDDEN_SIZE + 1][OUTPUT_SIZE + 1],
    int iterations) {

  float output_error = 0.0;

  for (int iteration = 0; iteration < iterations; iteration++) {
    float sum;
    int j, k;

    input_units[0] = 1.0;
    for (j = 1; j <= HIDDEN_SIZE; j++) {
      sum = 0.0;
      for (k = 0; k <= LAYERSIZE; k++) {
        sum += input_weights[k][j] * input_units[k];
      }
      hidden_units[j] = (1.0 / (1.0 + exp(-1 * sum)));
    }

    hidden_units[0] = 1.0;
    for (j = 1; j <= OUTPUT_SIZE; j++) {
      sum = 0.0;
      for (k = 0; k <= HIDDEN_SIZE; k++) {
        sum += hidden_weights[k][j] * hidden_units[k];
      }
      output_units[j] = (1.0 / (1.0 + exp(-sum)));
    }

    float o, t;
    output_error = 0.0;
    for (j = 1; j <= OUTPUT_SIZE; j++) {
      o = output_units[j];
      t = target[j];
      output_delta[j] = o * (1.0 - o) * (t - o);
      output_error += fabs(output_delta[j]);
    }

    float new_dw;
    hidden_units[0] = 1.0;
    for (j = 1; j <= OUTPUT_SIZE; j++) {
      for (k = 0; k <= HIDDEN_SIZE; k++) {
        new_dw = ((ETA * output_delta[j] * hidden_units[k]) +
                  (MOMENTUM * hidden_prev_weights[k][j]));
        hidden_weights[k][j] += new_dw;
        hidden_prev_weights[k][j] = new_dw;
      }
    }

    input_units[0] = 1.0;
    for (j = 1; j <= HIDDEN_SIZE; j++) {
      for (k = 0; k <= LAYERSIZE; k++) {
        new_dw = ((ETA * hidden_delta[j] * input_units[k]) +
                  (MOMENTUM * input_prev_weights[k][j]));
        input_weights[k][j] += new_dw;
        input_prev_weights[k][j] = new_dw;
      }
    }
  }
  return (output_error);
  printf("%f\n", output_error);
}
