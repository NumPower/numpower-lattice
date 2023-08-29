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
     * @param array $shape
     * @param ?int $batchSize
     * @param ?string|null $name
     */
    public function __construct(array $shape, ?int $batchSize = NULL, ?string $name = NULL) {
        $this->inputShape = $shape;
        $this->batchSize = $batchSize;
        ($name) ? $this->setName($name) : $this->setName("input_". substr(uniqid(), -4));
        parent::__construct();
    }
}