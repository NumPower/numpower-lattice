<?php

namespace NumPower\Lattice\Core\Models;

use NumPower\Lattice\Core\Layers\ILayer;

interface IModel
{
    /**
     * @return ILayer[]
     */
    public function getLayers(): array;

    /**
     * @param string $file_path
     * @return void
     */
    public function save(string $file_path): void;
}
