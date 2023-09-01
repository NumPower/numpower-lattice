<?php

namespace NumPower\Lattice\Models;

use NumPower\Lattice\Core\Layers\ILayer;

class Stack extends Network
{

    /**
     * @param ILayer $layer
     * @return void
     */
    public function add(ILayer $layer): void
    {
        $this->layers[] = $layer;
    }

    /**
     * @return void
     */
    public function pop(): void
    {
        array_pop($this->layers);
    }
}
