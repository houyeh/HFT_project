#include "qat12_stream_impl.hpp"

void qat12_scalar_hs_pipeline(
    QAT_SCALAR_INPUTS,
    qat_acc_t& logit0, qat_acc_t& logit1, qat_acc_t& logit2,
    qat_action_t& action
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_none port=logit0
#pragma HLS INTERFACE ap_none port=logit1
#pragma HLS INTERFACE ap_none port=logit2
#pragma HLS INTERFACE ap_none port=action
#pragma HLS PIPELINE II=1
    QAT_PACK_INPUTS;
    qat_compute(x, logit0, logit1, logit2, action);
}
