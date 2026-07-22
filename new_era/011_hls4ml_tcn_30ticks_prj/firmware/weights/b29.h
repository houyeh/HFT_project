//Numpy array shape [3]
//Min -0.056079536676
//Max 0.104687280953
//Number of zeros 0

#ifndef B29_H_
#define B29_H_

#ifndef __SYNTHESIS__
fc_bias_t b29[3];
#else
fc_bias_t b29[3] = {0.1046872810, -0.0345534422, -0.0560795367};

#endif

#endif
