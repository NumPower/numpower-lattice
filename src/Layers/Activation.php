<?php

namespace NumPower\Lattice\Layers;

use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Layers\Layer;

class Activation extends Layer
{
    /**
     * @var IActivation
     */
    public IActivation $activation;

    /**
     * @param IActivation $activation
     * @param string|null $name
     */
    public function __construct(IActivation $activation, ?string $name = null)
    {
        $this->activation = $activation;
        $this->trainable = false;
        ($name) ? $this->setName($name) : $this->setName("activation_" . substr(uniqid(), -4));
    }

    /**
     * @param int[] $inputShape
     * @return void
     */
    public function build(array $inputShape): void
    {
        $this->built = true;
        $this->setInputShape($inputShape);
    }
}
