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
    public function __construct(IActivation $activation, ?string $name = NULL)
    {
        $this->activation = $activation;
        $this->trainable = False;
        ($name) ? $this->setName($name) : $this->setName("activation_".substr(uniqid(), -4));
    }

    /**
     * @param array $inputShape
     * @return void
     */
    public function build(array $inputShape)
    {
        $this->built = True;
        $this->setInputShape($inputShape);
    }
}