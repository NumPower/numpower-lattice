<?php

namespace NumPower\Lattice\Utils;

use \NDArray as nd;
use NumPower\Lattice\Core\Layers\ILayer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

class LayerUtils
{
    /**
     * @param ILayer $layer
     * @param $positions
     * @param bool $showTrainable
     * @param int $nested_level
     * @return void
     */
    private static function printLayerSummary(ILayer $layer, $positions, bool $showTrainable = true, int $nested_level = 0)
    {
        $output_shape_str = "(";
        $output_shape = $layer->generateOutputShape();
        foreach ($output_shape as $idx => $s) {
            if ($idx == 0 && count($output_shape) > 1) {
                $output_shape_str .= $s;
                continue;
            }
            if (count($output_shape) == 1) {
                $output_shape_str .= "NULL, " . $s;
                continue;
            }
            $output_shape_str .= ',' . $s;
        }
        $output_shape_str .= ")";
        $name = $layer->getName();
        $class_name = explode("\\", $layer::class);
        $class_name = end($class_name);
        if (!$layer->built()) {
            $params = "0 (unused)";
        } else {
            $params = $layer->countParams();
        }
        $fields = [$name . " (" . $class_name . ")", $output_shape_str, $params];

        if ($showTrainable) {
            ($layer->isTrainable()) ? $fields[] = "Y" : $fields[] = "N";
        }
        self::printRow($fields, $positions, $nested_level);
    }

    /**
     * @param $fields
     * @param $positions
     * @param $nested_level
     * @return void
     */
    private static function printRow($fields, $positions, $nested_level = 0)
    {
        $left_to_print = $fields;
        $field_id = 0;
        $line = "";
        for ($col = 0; $col < count($left_to_print); $col++) {
            if ($col > 0) {
                $start_pos = $positions[$col - 1];
            } else {
                $start_pos = 0;
            }

            $end_pos = $positions[$col];

            (count($left_to_print) - 1 != $col) ? $space = 2 : $space = 0;
            $cutoff = $end_pos - $start_pos - $space;

            $fit_into_line = $left_to_print[$col];
            if ($col != count($positions) - 1) {
                $cutoff -= 1;
            }
            if ($col == 0) {
                $cutoff -= $nested_level;
            }

            if ($col == 0) {
                $line .= str_repeat("|", 1 + $nested_level) . " ";
            }

            $line .= $fit_into_line;
            $line .= str_repeat("\t", 1);
        }
        $line .= str_repeat(" |", 1 + $nested_level);
        print($line . "\n");
    }

    /**
     * @param Model $model
     * @param bool $showTrainable
     * @return void
     */
    public static function printSummary(Model $model, bool $showTrainable = true)
    {
        $line_length = 65;
        $positions = [0.45, 0.85, 1.0];
        $to_display = ["Layer (type)", "Output Shape", "Param #"];

        if ($showTrainable) {
            $line_length += 11;
            $positions[] = $line_length;
            $to_display[] = "Trainable";
        }

        print("Model: " . $model->getName() . "\n");
        print(str_repeat("-", $line_length) . "\n");
        self::printRow($to_display, $positions);
        print(str_repeat("=", $line_length) . "\n");
        foreach ($model->getLayers() as $layer) {
            self::printLayerSummary($layer, $positions, $showTrainable);
        }
        print(str_repeat("=", $line_length) . "\n");

        $total_params = 0;
        foreach ($model->getLayers() as $layer) {
            $total_params += $layer->countParams();
        }
        print("Total Params: " . $total_params . "\n");

        $total_params = 0;
        foreach ($model->getLayers() as $layer) {
            $total_params += $layer->countTrainableParams();
        }
        print("Total Params (Trainable): " . $total_params . "\n");

        $total_params = 0;
        foreach ($model->getLayers() as $layer) {
            $total_params += $layer->countNonTrainableParams();
        }
        print("Total Params (Non-Trainable): " . $total_params . "\n");
        print(str_repeat("-", $line_length) . "\n");
    }

    /**
     * @param Variable[] $weights
     * @return int
     */
    public static function countParams(array $weights): int
    {
        $w_shapes = [];
        foreach ($weights as $weight) {
            $w_shapes[] = $weight->getShape();
        }
        $prod = 0;
        foreach ($w_shapes as $shape) {
            $prod += nd::prod($shape);
        }
        return (int)$prod;
    }
}
