from imgaug import augmenters as iaa

def apply_augmentation(images):
    seq = iaa.Sequential([
        # Blur
        iaa.SomeOf((0, 1), [
            iaa.GaussianBlur(sigma=(1., 2.5)),
            iaa.AverageBlur(k=(2, 5))
        ]),
        # Rotation and padding
        iaa.SomeOf((0,1), [
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        ]),
        # Light
        iaa.SomeOf((0,2), [
            iaa.GammaContrast((0.5, 2.0)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
            iaa.Multiply((0.5, 1.5)),
            iaa.ContrastNormalization((0.5, 1.5))
        ]),
        # Noise
        iaa.SomeOf((0, 1), [
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
            iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255)),
            iaa.SaltAndPepper(0.02),
        ])
    ])

    image = seq(images=images)

    return image