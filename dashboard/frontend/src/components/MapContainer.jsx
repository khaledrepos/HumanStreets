import React from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import { GeoJsonLayer } from '@deck.gl/layers';

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
    const [geoJsonData, setGeoJsonData] = React.useState(null);

    React.useEffect(() => {
        fetch('http://localhost:8000/segmentations')
            .then(resp => resp.json())
            .then(data => {
                setGeoJsonData(data);
                console.log("Loaded segments:", data?.features?.length);
            })
            .catch(err => console.error("Failed to load segments:", err));
    }, []);

    const layers = [
        new GeoJsonLayer({
            id: 'sidewalk-layer',
            data: geoJsonData,
            pickable: true,
            stroked: false,
            filled: true,
            extruded: false,
            getFillColor: [50, 205, 50, 150], // Lime Green with transparency
            getLineColor: [0, 0, 0, 0],
            opacity: 0.8
        })
    ];

    const [viewState, setViewState] = React.useState(INITIAL_VIEW_STATE);

    const flyToSegments = () => {
        fetch('http://localhost:8000/segmentations/centroid')
            .then(resp => resp.json())
            .then(data => {
                if (data.latitude && data.longitude) {
                    setViewState({
                        ...viewState,
                        latitude: data.latitude,
                        longitude: data.longitude,
                        zoom: 18,
                        transitionDuration: 2000
                    });
                }
            })
            .catch(err => console.error("Failed to fetch centroid:", err));
    };

    return (
        <div className="main-content" style={{ position: 'relative' }}>
            <DeckGL
                initialViewState={undefined}
                viewState={viewState}
                onViewStateChange={({ viewState }) => setViewState(viewState)}
                controller={true}
                layers={layers}
            >
                <Map
                    mapStyle={MAP_STYLE}
                />
            </DeckGL>
            <button
                onClick={flyToSegments}
                style={{
                    position: 'absolute',
                    top: '20px',
                    right: '20px',
                    padding: '10px 20px',
                    background: '#ffffff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                    zIndex: 1000,
                    fontWeight: 'bold',
                    color: '#333'
                }}
            >
                Fly to Segments
            </button>
        </div>
    );
}

export default MapContainer;
