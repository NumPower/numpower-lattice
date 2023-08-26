<?php

namespace NumPower\Lattice\Initializers;

abstract class Initializer implements IInitializer
{
    protected array $shape;

    public function setShape(array $shape) {
        $this->shape = $shape;
    }
}