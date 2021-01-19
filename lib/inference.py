import torch
import torch.nn.functional as F


def inference_on_stack(stack_of_slices, model, slice_thickness_mm, pixel_diameter_mm=0.64):
    n_slices, colour_channels, y_in, x_in = stack_of_slices.size()
    model = model.eval()

    with torch.no_grad():
        segmented_stack = model(stack_of_slices)

        # We need to go from n_slices * C * H * W -> <BS=1> * C * n_slices * H * W
        segmented_stack_scaled = segmented_stack.transpose(0, 1).unsqueeze(0)

        #  If we have a slice-thickness of pixel_diameter we are already isotropic
        if slice_thickness_mm != pixel_diameter_mm:
            z_out = round(slice_thickness_mm / pixel_diameter_mm)

            segmented_stack_scaled = F.interpolate(segmented_stack_scaled,
                                                   size=(z_out, y_in, x_in),
                                                   mode='trilinear')

        # Remove the dummy batch size dimension
        segmented_stack_scaled = segmented_stack_scaled.squeeze(0)

        return segmented_stack, segmented_stack_scaled
