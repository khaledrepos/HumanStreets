import React from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';

const INITIAL_VIEW_STATE = {
    latitude: 40.7128,
    longitude: -74.0060,
    zoom: 12,
    pitch: 0,
    bearing: 0
};

// Use a public CARTO style to avoid needing an API key for the demo
const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

function MapContainer() {
    const layers = [
        // Add DeckGL layers here in the future
    ];

    return (
        <div className="main-content" style={{ position: 'relative' }}>
            <DeckGL
                initialViewState={INITIAL_VIEW_STATE}
                controller={true}
                layers={layers}
            >
                <Map
                    mapStyle={MAP_STYLE}
                />
            </DeckGL>
        </div>
    );
}

export default MapContainer;
