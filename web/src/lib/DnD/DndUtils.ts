import { defaultDropAnimationSideEffects, type DropAnimation } from "@dnd-kit-svelte/core";
import { KeyboardSensor, PointerSensor } from "@dnd-kit-svelte/svelte";

export const sensors = [KeyboardSensor, PointerSensor];

export const dropAnimation: DropAnimation = {
    sideEffects: defaultDropAnimationSideEffects({
        styles: {
            active: {
                opacity: '0.5',
            }
        }
    })
};