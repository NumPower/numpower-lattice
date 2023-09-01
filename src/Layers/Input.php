<?php

namespace NumPower\Lattice\Layers;

use NumPower\Lattice\Core\Layers\Layer;

class Input extends Layer
{
    /**
     * @var int|null
     */
    public ?int $batchSize;

    /**
     * @param int[] $shape
     * @param int|null $batchSize
     * @param string|null $name
     */
    public function __construct(array $shape, ?int $batchSize = null, ?string $name = null)
    {
        $this->inputShape = $shape;
        $this->batchSize = $batchSize;
        ($name) ? $this->setName($name) : $this->setName("input_" . substr(uniqid(), -4));
        $this->trainable = false;
        $this->built = false;
    }
}
